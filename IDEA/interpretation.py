from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from captum.attr import IntegratedGradients


# ============================================================
# Utilities
# ============================================================

def _safe_name(name):
    """Make a string safe for use as a file name."""
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))


# ============================================================
# Forward function for IDEA-I
# ============================================================

class RealSTForward(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        # 与模型输入保持一致
        x_log = torch.log1p(x).float()

        # Encoder
        q_m, q_v, dist, latent = self.model.vae_encoder(x_log)

        # 可解释分析使用 posterior mean，避免随机采样
        z = q_m

        n_spots = x.shape[0]

        # real ST 对应 batch = 0
        batch = torch.zeros(
            (n_spots, 1),
            device=x.device,
            dtype=x.dtype
        )

        # 当前 DeconvHead 实际并不使用下面三个变量
        library = torch.zeros(
            (n_spots, 1),
            device=x.device,
            dtype=x.dtype
        )

        x_pos = torch.zeros(
            (n_spots, 1),
            device=x.device,
            dtype=x.dtype
        )

        y_pos = torch.zeros(
            (n_spots, 1),
            device=x.device,
            dtype=x.dtype
        )

        pred = self.model.deconv(
            z,
            batch,
            library,
            x_pos,
            y_pos
        )

        return pred


# ============================================================
# Integrated Gradients
# ============================================================

def explain_celltypes(
    ideac_model,
    top_n=30,
    n_steps=50,
    output_dir=None,
):

    model = ideac_model.model
    device = ideac_model.device

    model.eval()

    gene_names = np.asarray(
        ideac_model.st_ad.var_names
    )

    celltype_names = ideac_model.celltype_names

    # ==========================================
    # ONLY REAL ST DATA
    # ==========================================
    data_st_set = DataLoader(
        ideac_model.st_set,
        batch_size=1,
        shuffle=False
    )

    forward_model = RealSTForward(model).to(device)
    forward_model.eval()

    ig = IntegratedGradients(forward_model)

    all_results = []
    top_results = []

    for celltype_idx, celltype in enumerate(celltype_names):

        print(
            f"[IDEA-I] {celltype_idx + 1}/"
            f"{len(celltype_names)}: {celltype}"
        )

        gene_contribution = torch.zeros(
            len(gene_names),
            device=device,
            dtype=torch.float32
        )

        for x, pos, spot_idx in tqdm(
            data_st_set,
            desc=f"[IG] {celltype}"
        ):

            # ----------------------------------
            # x here is REAL ST only
            # ----------------------------------

            x = x.squeeze(0).to(
                device=device,
                dtype=torch.float32
            )

            baseline = torch.zeros_like(x)

            attribution = ig.attribute(
                inputs=x,
                baselines=baseline,
                target=celltype_idx,
                n_steps=n_steps,
                internal_batch_size=x.shape[0],
            )

            # 当前 block 对各 gene 的贡献求和
            gene_contribution += (
                attribution
                .sum(dim=0)
                .detach()
            )

        contribution = (
            gene_contribution
            .cpu()
            .numpy()
        )

        result = pd.DataFrame({
            "celltype": celltype,
            "gene": gene_names,
            "contribution": contribution,
        })

        result["abs_contribution"] = (
            result["contribution"].abs()
        )

        result = result.sort_values(
            "abs_contribution",
            ascending=False
        ).reset_index(drop=True)

        result["rank"] = (
            np.arange(len(result)) + 1
        )

        top_result = result.head(top_n).copy()

        all_results.append(result)
        top_results.append(top_result)

    all_df = pd.concat(
        all_results,
        ignore_index=True
    )

    top_df = pd.concat(
        top_results,
        ignore_index=True
    )

    genes = pd.unique(
        top_df["gene"]
    ).tolist()

    return {
        "all": all_df,
        "top": top_df,
        "genes": genes,
    }

# ============================================================
# Spatial statistics
# ============================================================

def calculate_spatial_statistics(
    adata,
    genes,
    coord_key=None,
    k=6,
    permutations=999,
    calculate_geary=False,
):
    """
    Calculate spatial autocorrelation for candidate genes.

    Parameters
    ----------
    adata
        Spatial AnnData.
    genes
        Candidate genes.
    coord_key
        Spatial coordinate key.
    k
        Number of spatial neighbors.
    permutations
        Number of permutations.
    calculate_geary
        Whether to additionally calculate Geary's C.
    """

    import libpysal
    from esda import Moran, Geary

    # --------------------------------------------------------
    # Spatial coordinates
    # --------------------------------------------------------

    if coord_key is None:

        if "location" in adata.obsm:
            coord_key = "location"

        elif "spatial" in adata.obsm:
            coord_key = "spatial"

        elif "X_spatial" in adata.obsm:
            coord_key = "X_spatial"

        else:
            raise KeyError(
                "No spatial coordinates found in "
                "adata.obsm['location'], "
                "adata.obsm['spatial'], or "
                "adata.obsm['X_spatial']."
            )

    coords = np.asarray(
        adata.obsm[coord_key]
    )

    # --------------------------------------------------------
    # KNN graph
    # --------------------------------------------------------

    w = libpysal.weights.KNN.from_array(
        coords,
        k=k
    )

    w.transform = "r"

    genes = [
        gene
        for gene in genes
        if gene in adata.var_names
    ]

    results = []

    for gene in tqdm(
        genes,
        desc="[IDEA-I] Moran"
    ):

        x = adata[:, gene].X

        if not isinstance(x, np.ndarray):
            x = x.toarray()

        x = np.asarray(
            x
        ).flatten()

        mi = Moran(
            x,
            w,
            permutations=permutations
        )

        row = {
            "gene": gene,
            "moran_I": mi.I,
            "moran_p": mi.p_sim,
        }

        if calculate_geary:

            gc = Geary(
                x,
                w,
                permutations=permutations
            )

            row.update({
                "geary_C": gc.C,
                "geary_p": gc.p_sim,
            })

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# Differential expression
# ============================================================

def calculate_de(
    adata,
    genes,
    groupby,
    normalize=True,
    method="wilcoxon",
    prefix=None,
):
    """
    Calculate DE statistics for candidate genes.

    For each gene, all group-specific DE results are returned.
    """

    import scanpy as sc

    adata = adata.copy()

    if groupby not in adata.obs:
        raise KeyError(
            f"{groupby!r} not found in adata.obs."
        )

    if normalize:

        sc.pp.normalize_total(
            adata,
            target_sum=1e4
        )

        sc.pp.log1p(
            adata
        )

    adata.obs[groupby] = (
        adata.obs[groupby]
        .astype("category")
    )

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        pts=True
    )

    gene_set = set(genes)

    groups = (
        adata.obs[groupby]
        .cat.categories
    )

    de_results = []

    for group in groups:

        names = (
            adata.uns[
                "rank_genes_groups"
            ]["names"][str(group)]
        )

        pvals = (
            adata.uns[
                "rank_genes_groups"
            ]["pvals"][str(group)]
        )

        pvals_adj = (
            adata.uns[
                "rank_genes_groups"
            ]["pvals_adj"][str(group)]
        )

        logfc = (
            adata.uns[
                "rank_genes_groups"
            ]["logfoldchanges"][str(group)]
        )

        for gene, p, padj, fc in zip(
            names,
            pvals,
            pvals_adj,
            logfc
        ):

            if gene in gene_set:

                de_results.append({
                    "gene": gene,
                    "group": group,
                    "pval": p,
                    "fdr": padj,
                    "logFC": fc,
                })

    df = pd.DataFrame(
        de_results
    )

    if df.empty:
        return df

    # --------------------------------------------------------
    # Prefix source
    # --------------------------------------------------------

    if prefix is not None:

        df = df.rename(
            columns={
                "group":
                    f"{prefix}_group",

                "pval":
                    f"{prefix}_pval",

                "fdr":
                    f"{prefix}_fdr",

                "logFC":
                    f"{prefix}_logFC",
            }
        )

    return df


# ============================================================
# Select best DE group for each gene
# ============================================================

def select_best_de(
    de_df,
    prefix,
):
    """
    Select the most significant group for each gene.
    """

    if de_df is None or de_df.empty:
        return pd.DataFrame(
            columns=["gene"]
        )

    fdr_col = f"{prefix}_fdr"

    best = (
        de_df
        .sort_values(
            fdr_col
        )
        .groupby(
            "gene",
            as_index=False
        )
        .first()
    )

    return best


# ============================================================
# Candidate gene validation
# ============================================================

def validate_candidate_genes(
    adata,
    genes,
    sc_ad=None,

    # --------------------------------------------------------
    # What to calculate
    # "moran", "logfc", or "both"
    # --------------------------------------------------------
    metrics="both",

    # --------------------------------------------------------
    # Resolution
    # "low"  : only scRNA-seq logFC
    # "high" : scRNA-seq logFC + spatial pre logFC
    # --------------------------------------------------------
    resolution="high",

    # --------------------------------------------------------
    # Grouping
    # --------------------------------------------------------
    sc_groupby="celltype",
    pre_groupby="pre",

    # spatial
    coord_key=None,
    k=6,
    permutations=999,
    calculate_geary=False,

    # DE
    normalize_sc=True,
    normalize_pre=True,
    de_method="wilcoxon",

    # thresholds
    moran_th=0.1,
    fdr_th=0.05,
    logfc_th=0.5,

    output_path=None,
):

    """
    Validate IDEA-I candidate genes.

    Parameters
    ----------
    metrics : {"moran", "logfc", "both"}
        Which statistics to calculate.

    resolution : {"low", "high"}
        low:
            calculate logFC only from sc_ad.

        high:
            calculate logFC from both sc_ad and
            spatial adata.obs[pre_groupby].

    sc_groupby
        Cell-type annotation column in sc_ad.obs.

    pre_groupby
        Predicted class / cell-type column in spatial adata.obs.
    """

    # ========================================================
    # 0. Check parameters
    # ========================================================

    metrics = metrics.lower()
    resolution = resolution.lower()

    if metrics not in {
        "moran",
        "logfc",
        "both",
    }:
        raise ValueError(
            "metrics must be one of "
            "{'moran', 'logfc', 'both'}."
        )

    if resolution not in {
        "low",
        "high",
    }:
        raise ValueError(
            "resolution must be "
            "'low' or 'high'."
        )

    calculate_moran = (
        metrics in {
            "moran",
            "both",
        }
    )

    calculate_logfc = (
        metrics in {
            "logfc",
            "both",
        }
    )

    # --------------------------------------------------------
    # Start from all candidate genes
    # --------------------------------------------------------

    result = pd.DataFrame({
        "gene": list(
            dict.fromkeys(genes)
        )
    })

    # ========================================================
    # 1. Moran's I
    # ========================================================

    if calculate_moran:

        spatial_df = (
            calculate_spatial_statistics(
                adata=adata,
                genes=genes,
                coord_key=coord_key,
                k=k,
                permutations=permutations,
                calculate_geary=calculate_geary,
            )
        )

        result = result.merge(
            spatial_df,
            on="gene",
            how="left",
        )

    # ========================================================
    # 2. scRNA-seq logFC
    # ========================================================

    if calculate_logfc:

        if sc_ad is None:

            raise ValueError(
                "sc_ad is required when "
                "metrics includes 'logfc'."
            )

        sc_de = calculate_de(
            adata=sc_ad,
            genes=genes,
            groupby=sc_groupby,
            normalize=normalize_sc,
            method=de_method,
            prefix="sc",
        )

        sc_best = select_best_de(
            sc_de,
            prefix="sc",
        )

        result = result.merge(
            sc_best,
            on="gene",
            how="left",
        )

    # ========================================================
    # 3. High-resolution spatial/pre logFC
    # ========================================================

    if (
        calculate_logfc
        and resolution == "high"
    ):

        if pre_groupby not in adata.obs:

            raise KeyError(
                f"resolution='high' requires "
                f"{pre_groupby!r} in adata.obs."
            )

        pre_de = calculate_de(
            adata=adata,
            genes=genes,
            groupby=pre_groupby,
            normalize=normalize_pre,
            method=de_method,
            prefix="pre",
        )

        pre_best = select_best_de(
            pre_de,
            prefix="pre",
        )

        result = result.merge(
            pre_best,
            on="gene",
            how="left",
        )

    # ========================================================
    # 4. Individual validation flags
    # ========================================================

    if calculate_moran:

        result["pass_moran"] = (
            result["moran_I"]
            > moran_th
        )

    if calculate_logfc:

        result["pass_sc_logfc"] = (
            (result["sc_fdr"] < fdr_th)
            &
            (
                result["sc_logFC"].abs()
                > logfc_th
            )
        )

    if (
        calculate_logfc
        and resolution == "high"
    ):

        result["pass_pre_logfc"] = (
            (result["pre_fdr"] < fdr_th)
            &
            (
                result["pre_logFC"].abs()
                > logfc_th
            )
        )

    # ========================================================
    # 5. Final selection
    # ========================================================

    if metrics == "moran":

        result["is_selected"] = (
            result["pass_moran"]
        )

    elif metrics == "logfc":

        if resolution == "low":

            result["is_selected"] = (
                result[
                    "pass_sc_logfc"
                ]
            )

        else:

            result["is_selected"] = (
                result[
                    "pass_sc_logfc"
                ]
                &
                result[
                    "pass_pre_logfc"
                ]
            )

    else:
        # both

        if resolution == "low":

            result["is_selected"] = (
                result["pass_moran"]
                &
                result["pass_sc_logfc"]
            )

        else:

            result["is_selected"] = (
                result["pass_moran"]
                &
                result["pass_sc_logfc"]
                &
                result["pass_pre_logfc"]
            )

    # ========================================================
    # 6. Optional score
    # ========================================================

    if metrics == "moran":

        result["score"] = (
            result["moran_I"]
        )

    elif metrics == "logfc":

        if resolution == "low":

            result["score"] = (
                result["sc_logFC"].abs()
                *
                (
                    -np.log10(
                        result["sc_fdr"]
                        .clip(
                            lower=1e-10
                        )
                    )
                )
            )

        else:

            result["score"] = (
                result[
                    "sc_logFC"
                ].abs()
                *
                result[
                    "pre_logFC"
                ].abs()
                *
                (
                    -np.log10(
                        result[
                            "sc_fdr"
                        ].clip(
                            lower=1e-10
                        )
                    )
                )
                *
                (
                    -np.log10(
                        result[
                            "pre_fdr"
                        ].clip(
                            lower=1e-10
                        )
                    )
                )
            )

    else:

        if resolution == "low":

            result["score"] = (
                result["moran_I"]
                *
                result[
                    "sc_logFC"
                ].abs()
                *
                (
                    -np.log10(
                        result[
                            "sc_fdr"
                        ].clip(
                            lower=1e-10
                        )
                    )
                )
            )

        else:

            result["score"] = (
                result["moran_I"]
                *
                result[
                    "sc_logFC"
                ].abs()
                *
                result[
                    "pre_logFC"
                ].abs()
                *
                (
                    -np.log10(
                        result[
                            "sc_fdr"
                        ].clip(
                            lower=1e-10
                        )
                    )
                )
                *
                (
                    -np.log10(
                        result[
                            "pre_fdr"
                        ].clip(
                            lower=1e-10
                        )
                    )
                )
            )

    # ========================================================
    # 7. Sort
    # ========================================================

    result = (
        result
        .sort_values(
            "score",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    result["rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    # ========================================================
    # 8. Save
    # ========================================================

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        result.to_csv(
            output_path,
            index=False
        )

    return result