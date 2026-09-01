from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from captum.attr import IntegratedGradients


# ============================================================
# Utilities
# ============================================================


def _safe_name(name):
    """Make a string safe for use as a file name."""
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))


def _knn_from_pos(pos: torch.Tensor, k: int):
    """Build KNN using only spots in the current spatial block."""
    if pos.ndim != 2:
        raise ValueError(f"pos must be 2D, got shape {tuple(pos.shape)}")

    n_spots = int(pos.shape[0])
    if n_spots < 2:
        raise ValueError("A spatial block must contain at least two spots.")

    k_eff = min(int(k), n_spots - 1)
    pos_np = pos.detach().cpu().numpy()

    nn_model = NearestNeighbors(n_neighbors=k_eff + 1).fit(pos_np)
    dists, indices = nn_model.kneighbors(pos_np)

    knn_idx = torch.as_tensor(
        indices[:, 1:],
        dtype=torch.long,
        device=pos.device,
    )
    knn_dist = torch.as_tensor(
        dists[:, 1:],
        dtype=torch.float32,
        device=pos.device,
    )

    return knn_idx, knn_dist


# ============================================================
# Differentiable forward for one spatial block
# ============================================================


class NicheBlockForward(nn.Module):
    """Differentiable surrogate used by Integrated Gradients.

    The whole spatial block is treated as ONE Captum example because KANA
    introduces dependencies between spots through the block-local KNN graph.

    For a target niche, the output is the sum (or mean) of that niche's
    classifier logits over spots assigned to that niche by the final Leiden
    clustering. The classifier is therefore used only as a differentiable
    surrogate of the final clustering result.
    """

    def __init__(
        self,
        model: nn.Module,
        knn_idx: torch.Tensor,
        knn_dist: torch.Tensor,
        target_mask: torch.Tensor,
        niche_idx: int,
        n_clusters: int,
        score_reduction: str = "sum",
        use_posterior_mean: bool = True,
    ):
        super().__init__()
        self.model = model
        self.niche_idx = int(niche_idx)
        self.n_clusters = int(n_clusters)
        self.score_reduction = str(score_reduction)
        self.use_posterior_mean = bool(use_posterior_mean)

        self.register_buffer("knn_idx", knn_idx.long())
        self.register_buffer("knn_dist", knn_dist.float())
        self.register_buffer("target_mask", target_mask.bool())

        if self.score_reduction not in {"sum", "mean"}:
            raise ValueError("score_reduction must be 'sum' or 'mean'.")

    def _forward_one(self, x: torch.Tensor) -> torch.Tensor:
        # x is raw/non-negative count-like input: [N_spots, N_genes]
        x_log = torch.log1p(x).float()

        # Niche encoder includes KANA, so KNN is part of the attribution path.
        _, q_m, _, _, latent = self.model.vae_encoder(
            x_log,
            self.knn_idx,
            self.knn_dist,
        )

        # Deterministic interpretation by default.
        z = q_m if self.use_posterior_mean else latent

        # Condition is intentionally NOT part of the niche prediction path.
        logits = self.model.domain_classifier(z)[:, : self.n_clusters]
        target_score = logits[self.target_mask, self.niche_idx]

        if target_score.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        if self.score_reduction == "mean":
            return target_score.mean()

        return target_score.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Captum sees the whole spatial block as one example:
        # input shape expected: [B, N_spots, N_genes].
        if x.ndim == 2:
            x = x.unsqueeze(0)

        if x.ndim != 3:
            raise ValueError(
                "NicheBlockForward expects [B, N_spots, N_genes], "
                f"got {tuple(x.shape)}."
            )

        scores = [self._forward_one(x_i) for x_i in x]
        return torch.stack(scores, dim=0)


# ============================================================
# Integrated Gradients for niche-associated genes
# ============================================================


def explain_niches(
    idean_model,
    top_n: int = 30,
    n_steps: int = 50,
    output_dir=None,
    use_posterior_mean: bool = True,
    score_reduction: str = "sum",
    attribution_scope: str = "all",
    normalize_by_target_spots: bool = False,
):
    """Identify niche-associated genes from real spatial data only.

    Parameters
    ----------
    idean_model
        Trained IDEANModel wrapper.

    top_n
        Number of genes retained for each niche.

    n_steps
        Number of Integrated Gradients integration steps.

    output_dir
        Optional directory for CSV outputs.

    use_posterior_mean
        If True, use q_m instead of a stochastic latent sample.

    score_reduction
        'sum' or 'mean' over the target-niche classifier logits in each block.
        'sum' most closely matches the original contribution aggregation logic.

    attribution_scope
        'all': sum attribution over every input spot in the block. This retains
        both direct and KANA-mediated neighbor contributions to the target niche.

        'target_spots': only aggregate attribution located on spots whose final
        Leiden label is the target niche.

    normalize_by_target_spots
        If True, divide each niche's final gene contribution by the number of
        target spots. Ranking within a niche is unchanged by this global scalar,
        but magnitudes become more comparable across niches.
    """

    if attribution_scope not in {"all", "target_spots"}:
        raise ValueError("attribution_scope must be 'all' or 'target_spots'.")

    model = idean_model.model
    device = idean_model.device
    model.eval()

    if idean_model.n_clusters is None:
        raise RuntimeError("The model has no n_clusters. Run/load IDEA-N training first.")

    if idean_model.pseudo_labels is None:
        raise RuntimeError(
            "Leiden niche labels are missing. IDEA-I for niches uses the final "
            "Leiden labels to define target spots."
        )

    # Optional soft warning: Stage 2 classifier is needed as the differentiable surrogate.
    if hasattr(idean_model, "history") and isinstance(idean_model.history, pd.DataFrame):
        if not idean_model.history.empty and "stage" in idean_model.history.columns:
            if not np.any(idean_model.history["stage"].to_numpy() == 2):
                warnings.warn(
                    "No Stage-2 history was found. Niche interpretation assumes the "
                    "domain classifier has been trained to reproduce the Leiden labels."
                )

    gene_names = np.asarray(idean_model.gene_names)
    n_genes = len(gene_names)
    n_clusters = int(idean_model.n_clusters)
    final_labels = np.asarray(idean_model.pseudo_labels, dtype=np.int64)

    if final_labels.shape[0] != int(idean_model.metadata["n_spots"]):
        raise ValueError(
            "pseudo_labels length does not match the number of spatial spots."
        )

    # Non-overlapping real-ST blocks are used so each spot is attributed once.
    data_st_set = DataLoader(
        idean_model.eval_set,
        batch_size=1,
        shuffle=False,
    )

    # [niche, gene]
    gene_contribution = torch.zeros(
        (n_clusters, n_genes),
        device=device,
        dtype=torch.float32,
    )
    target_spot_counts = np.zeros(n_clusters, dtype=np.int64)
    contributing_blocks = np.zeros(n_clusters, dtype=np.int64)

    for x, pos, global_idx, _condition_id in tqdm(
        data_st_set,
        desc="[IDEA-I niche] blocks",
    ):
        x = x.squeeze(0).to(device=device, dtype=torch.float32)
        pos = pos.squeeze(0).to(device=device, dtype=torch.float32)
        gi = global_idx.squeeze(0).cpu().numpy()

        if not torch.isfinite(x).all():
            raise FloatingPointError("Non-finite expression values found during explanation.")
        if torch.any(x < 0):
            raise ValueError(
                "Negative expression values found. The IDEA-N explanation path "
                "expects the same non-negative input space used by the model."
            )

        # Build neighborhoods only within the current block.
        if hasattr(idean_model, "_build_batch_geometry"):
            knn_idx, knn_dist, _, _ = idean_model._build_batch_geometry(
                pos,
                need_pairwise_distance=False,
            )
        else:
            knn_idx, knn_dist = _knn_from_pos(pos, idean_model.knn_k)

        block_labels = final_labels[gi]
        niches_in_block = np.unique(block_labels)

        # Captum input has a leading batch dimension of 1 so that the entire
        # spatial block stays intact at every IG interpolation point.
        inputs = x.unsqueeze(0)
        baseline = torch.zeros_like(inputs)

        for niche_idx in niches_in_block:
            niche_idx = int(niche_idx)
            if niche_idx < 0 or niche_idx >= n_clusters:
                continue

            target_mask_np = block_labels == niche_idx
            n_target = int(target_mask_np.sum())
            if n_target == 0:
                continue

            target_spot_counts[niche_idx] += n_target
            contributing_blocks[niche_idx] += 1

            target_mask = torch.as_tensor(
                target_mask_np,
                dtype=torch.bool,
                device=device,
            )

            forward_model = NicheBlockForward(
                model=model,
                knn_idx=knn_idx,
                knn_dist=knn_dist,
                target_mask=target_mask,
                niche_idx=niche_idx,
                n_clusters=n_clusters,
                score_reduction=score_reduction,
                use_posterior_mean=use_posterior_mean,
            ).to(device)
            forward_model.eval()

            ig = IntegratedGradients(forward_model)

            attribution = ig.attribute(
                inputs=inputs,
                baselines=baseline,
                target=None,
                n_steps=n_steps,
                # Important: keep one whole spatial block as one Captum example.
                internal_batch_size=1,
            )

            # [1, N_spots, N_genes] -> [N_spots, N_genes]
            attribution = attribution.squeeze(0)

            if attribution_scope == "target_spots":
                block_gene_contribution = attribution[target_mask].sum(dim=0)
            else:
                # Includes genes on neighboring spots that influence target spots
                # through KANA, which is the faithful full-model attribution.
                block_gene_contribution = attribution.sum(dim=0)

            gene_contribution[niche_idx] += block_gene_contribution.detach()

    contribution_np = gene_contribution.detach().cpu().numpy()

    if normalize_by_target_spots:
        denom = np.maximum(target_spot_counts, 1)[:, None]
        contribution_np = contribution_np / denom

    all_results = []
    top_results = []

    for niche_idx in range(n_clusters):
        contribution = contribution_np[niche_idx]

        result = pd.DataFrame(
            {
                "niche": str(niche_idx),
                "gene": gene_names,
                "contribution": contribution,
            }
        )
        result["abs_contribution"] = result["contribution"].abs()
        result = result.sort_values(
            "abs_contribution",
            ascending=False,
        ).reset_index(drop=True)
        result["rank"] = np.arange(len(result)) + 1
        result["n_target_spots"] = int(target_spot_counts[niche_idx])
        result["n_blocks"] = int(contributing_blocks[niche_idx])

        top_result = result.head(top_n).copy()

        all_results.append(result)
        top_results.append(top_result)

    all_df = pd.concat(all_results, ignore_index=True)
    top_df = pd.concat(top_results, ignore_index=True)
    genes = pd.unique(top_df["gene"]).tolist()

    # Optional output files.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_df.to_csv(output_dir / "all_niche_gene_contributions.csv", index=False)
        top_df.to_csv(output_dir / "top_niche_gene_contributions.csv", index=False)
        pd.DataFrame({"gene": genes}).to_csv(
            output_dir / "candidate_niche_genes.csv",
            index=False,
        )

        for niche_idx in range(n_clusters):
            niche_dir = output_dir / f"niche_{_safe_name(niche_idx)}"
            niche_dir.mkdir(parents=True, exist_ok=True)

            niche_all = all_df[all_df["niche"] == str(niche_idx)]
            niche_top = top_df[top_df["niche"] == str(niche_idx)]

            niche_all.to_csv(
                niche_dir / "gene_contributions.csv",
                index=False,
            )
            niche_top.to_csv(
                niche_dir / f"top_{top_n}_genes.csv",
                index=False,
            )

    return {
        "all": all_df,
        "top": top_df,
        "genes": genes,
        "target_spot_counts": target_spot_counts,
        "contributing_blocks": contributing_blocks,
    }