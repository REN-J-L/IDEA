from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scvi.distributions import ZeroInflatedNegativeBinomial
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.autograd import Function
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FCResBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0, activation="leakyrelu", norm="layer"):
        super().__init__()
        hidden_dim = hidden_dim or dim

        if norm == "layer":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        elif norm == "batch":
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError("norm must be 'layer' or 'batch'.")

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError("Unsupported activation.")

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        # Retained because this parameter existed in the original implementation.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.act(out)
        out = self.norm2(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.act(out)
        out = self.norm1(out)
        out = self.dropout(out)
        # out = self.norm1(x)
        # out = self.fc1(out)
        # out = self.act(out)
        # out = self.dropout(out)
        # out = self.norm2(out)
        # out = self.fc2(out)
        # out = self.dropout(out)
        return identity + out


class AttentionKernelPoolingEfficient(nn.Module):
    """KANA local kernel-attention aggregation."""

    def __init__(self, dim, hidden_dim=64, learnable_sigma=True):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_sigma", torch.zeros(1))

        # Retained for state compatibility with the research code.
        self.logit_lambda = nn.Parameter(torch.zeros(1))
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.rec = nn.Parameter(torch.zeros(1))
        self.gate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gamma_k = nn.Parameter(torch.zeros(1))
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z, knn_idx, knn_dist):
        n, k = knn_idx.shape
        z_j = z[knn_idx]

        first_linear = self.score_net[0]
        activation = self.score_net[1]
        output_linear = self.score_net[2]
        feature_dim = first_linear.in_features // 2

        z_i_flat = z.reshape(n, -1)
        z_j_flat = z_j.reshape(n, k, -1)
        if z_i_flat.shape[-1] != feature_dim:
            raise ValueError(
                f"Flattened feature dimension {z_i_flat.shape[-1]} does not match "
                f"score_net expectation {feature_dim}."
            )

        # Equivalent to Linear(cat([z_i, z_j])) without materializing [N*K, 2D].
        weight_i = first_linear.weight[:, :feature_dim]
        weight_j = first_linear.weight[:, feature_dim:]
        hidden_i = F.linear(z_i_flat, weight_i, first_linear.bias).unsqueeze(1)
        hidden_j = F.linear(z_j_flat, weight_j, bias=None)
        attn_logits = output_linear(activation(hidden_i + hidden_j)).squeeze(-1)

        sigma = torch.exp(self.log_sigma) + 1e-6
        kernel_weight = torch.exp(-(knn_dist ** 2) / (2 * sigma ** 2))
        kernel_weight = kernel_weight / (kernel_weight.sum(dim=1, keepdim=True) + 1e-8)
        z_smooth = torch.bmm(kernel_weight.unsqueeze(1), z_j_flat).squeeze(1).reshape_as(z)

        alpha = F.softmax(attn_logits, dim=1)
        z_attn = torch.bmm(alpha.unsqueeze(1), z_j_flat).squeeze(1).reshape_as(z)

        # Preserve the current research-code fusion exactly.
        return self.gamma * z + self.gamma_a * z_attn + self.gamma_k * z_smooth


class NicheEncoder(nn.Module):
    """Shared encoder. Condition is intentionally not supplied here."""

    def __init__(self, input_size, hidden_size=512, latent_size=128, dropout=0.0, var_eps=1e-4):
        super().__init__()
        self.vae_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=0.001),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=0.001),
            nn.Dropout(dropout),
        )
        self.kana = AttentionKernelPoolingEfficient(input_size)
        self.res1 = FCResBlock(hidden_size, latent_size)
        self.res2 = FCResBlock(hidden_size, latent_size)
        self.res3 = FCResBlock(hidden_size, latent_size)
        self.mean_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=0.001),
            nn.Linear(hidden_size, hidden_size),
        )
        self.var_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=0.001),
            nn.Linear(hidden_size, hidden_size),
        )
        self.var_eps = var_eps

    def forward(self, x, knn_idx, knn_dist):
        h = self.kana(x, knn_idx, knn_dist)


        h = self.vae_encoder(h)
        # h = self.kana(h, knn_idx, knn_dist)

        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        q_m = self.mean_encoder(h)
        q_v = torch.exp(self.var_encoder(h)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()
        return h, q_m, q_v, dist, latent


class NicheDecoder(nn.Module):
    """ZINB decoder with optional slice/platform conditioning.

    The conditional branch preserves the uploaded batch implementation:
    concatenate [z, condition] -> conditional MLP, then repeat once more.
    """

    def __init__(
        self,
        hidden_size,
        latent_size,
        output_size,
        n_conditions=1,
        use_condition=False,
        eps=1e-3,
    ):
        super().__init__()
        self.n_conditions = int(n_conditions)
        self.use_condition = bool(use_condition)

        self.res1 = FCResBlock(hidden_size, latent_size)
        self.res2 = FCResBlock(hidden_size, latent_size)
        self.res3 = FCResBlock(hidden_size, latent_size)

        if self.use_condition:
            self.condition_decoder = nn.Sequential(
                nn.Linear(hidden_size + self.n_conditions, hidden_size),
                nn.LeakyReLU(),
                nn.LayerNorm(hidden_size, eps=eps),

                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.LayerNorm(hidden_size, eps=eps),
            )
        else:
            self.condition_decoder = None

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.Linear(hidden_size,output_size),
            nn.Softmax(dim=-1),
        )

        self.px_dropout_decoder = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.Linear(hidden_size,output_size),
            )
        self.px_r_decoder = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.Linear(hidden_size,output_size),
            )

        if self.n_conditions > 1:
            self.px_r = torch.nn.Parameter(torch.randn(output_size,self.n_conditions))
        else:
            self.register_parameter("px_r",None)


    def forward(self, z, library, condition=None):
        z = self.res1(z)
        z = self.res2(z)
        z = self.res3(z)

        if self.use_condition:
            if condition is None:
                raise ValueError("condition is required when use_condition=True.")
            if condition.ndim != 2 or condition.shape[0] != z.shape[0]:
                raise ValueError(
                    f"condition must have shape [N, {self.n_conditions}], got {tuple(condition.shape)}."
                )
            if condition.shape[1] != self.n_conditions:
                raise ValueError(
                    f"Expected {self.n_conditions} condition columns, got {condition.shape[1]}."
                )

            z = self.condition_decoder(torch.cat([z, condition], dim=1))
            z = self.condition_decoder(torch.cat([z, condition], dim=1))

        px = z
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        if self.n_conditions == 1:
            px_r = self.px_r_decoder(px)
        else:
            if condition is None:
                raise ValueError(
                    "condition is required for " "condition-specific px_r when " "n_conditions > 1."
                )
            px_r = F.linear(condition,self.px_r)
        px_r = torch.exp(px_r)
        return px_scale, px_r, px_rate, px_dropout, px


class DomainClassifier(nn.Module):
    """Niche classifier. Condition is intentionally not supplied here."""

    def __init__(self, hidden_size, latent_size, max_cluster=100):
        super().__init__()
        self.res1 = FCResBlock(hidden_size, latent_size)
        self.res2 = FCResBlock(hidden_size, latent_size)
        self.res3 = FCResBlock(hidden_size, latent_size)
        self.z_cluster = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size,eps=0.001),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, max_cluster),
        )

    def forward(self, z):
        z = self.res1(z)
        z = self.res2(z)
        z = self.res3(z)
        return self.z_cluster(z)


class GradientReversalFunction(Function):
    """Gradient reversal layer used for optional slice-domain adversarial training.

    Forward is the identity. During backpropagation, the gradient entering the
    shared representation is multiplied by ``-alpha``. This lets the slice
    discriminator minimize slice-classification loss while the encoder is
    simultaneously encouraged to remove slice-specific information.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class SliceDomainClassifier(nn.Module):
    """Predict slice/platform identity from the shared latent representation."""

    def __init__(self, hidden_size, latent_size, n_domains, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.BatchNorm1d(
                latent_size,
                eps=1e-3,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.LeakyReLU(),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(
                latent_size,
                eps=1e-3,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size, latent_size),
            nn.BatchNorm1d(
                latent_size,
                eps=1e-3,
                momentum=0.01,
                affine=True,
                track_running_stats=True,
            ),
            nn.LeakyReLU(),
            nn.Linear(latent_size, n_domains),
        )

    def forward(self, feature, alpha=1.0):
        reverse_feature = GradientReversalFunction.apply(feature, alpha)
        reverse_feature = self.features(reverse_feature.reshape(reverse_feature.shape[0], -1))
        return self.domain_classifier(reverse_feature)


class NicheNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=512,
        latent_size=128,
        dropout=0.0,
        var_eps=1e-4,
        max_cluster=100,
        n_conditions=1,
        use_condition=False,
        use_domain_adversarial=False,
    ):
        super().__init__()
        self.use_condition = bool(use_condition)
        self.use_domain_adversarial = bool(use_domain_adversarial)
        self.n_conditions = int(n_conditions)

        self.vae_encoder = NicheEncoder(
            input_size,
            hidden_size,
            latent_size,
            dropout=dropout,
            var_eps=var_eps,
        )
        self.vae_decoder = NicheDecoder(
            hidden_size,
            latent_size,
            input_size,
            n_conditions=n_conditions,
            use_condition=use_condition,
        )
        self.domain_classifier = DomainClassifier(
            hidden_size,
            latent_size,
            max_cluster=max_cluster,
        )
        self.batch_classifier = (
            SliceDomainClassifier(
                hidden_size=hidden_size,
                latent_size=latent_size,
                n_domains=self.n_conditions,
                dropout=dropout,
            )
            if self.use_domain_adversarial
            else None
        )

    def forward(
        self,
        x,
        knn_idx,
        knn_dist,
        condition=None,
        use_posterior_mean=False,
        domain_alpha=0.0,
    ):
        x_log = torch.log1p(x).float()
        library = torch.log(x.sum(1).clamp_min(1e-8)).unsqueeze(1).float()
        h, q_m, q_v, dist, latent = self.vae_encoder(x_log, knn_idx, knn_dist)
        z = q_m if use_posterior_mean else latent
        px_scale, px_r, px_rate, px_dropout, px_dec = self.vae_decoder(
            z, library, condition=condition
        )
        soft_cluster = self.domain_classifier(z)
        batch_logits = (
            self.batch_classifier(z, alpha=domain_alpha)
            if self.batch_classifier is not None
            else None
        )
        return (
            q_m,
            q_v,
            dist,
            z,
            px_scale,
            px_r,
            px_rate,
            px_dropout,
            px_dec,
            h,
            soft_cluster,
            batch_logits,
        )

    def encode(self, x, knn_idx, knn_dist, use_posterior_mean=False):
        x_log = torch.log1p(x).float()
        _, q_m, _, _, latent = self.vae_encoder(x_log, knn_idx, knn_dist)
        return q_m if use_posterior_mean else latent


# -----------------------------------------------------------------------------
# Losses and geometry utilities
# -----------------------------------------------------------------------------

def reconst_loss(px_rate, px_r, px_dropout, px_scale, x):
    px = ZeroInflatedNegativeBinomial(
        mu=px_rate,
        theta=px_r,
        zi_logits=px_dropout,
        scale=px_scale,
    )
    return -px.log_prob(x).sum(-1)



def domain_cross_entropy_loss(pred_logits, targets, label_smoothing=0.1, reduction="mean"):
    """Label-smoothed slice/domain classification loss.

    The discriminator minimizes this loss. Because its input passes through a
    gradient-reversal layer, the encoder receives the opposite gradient and is
    encouraged to make slice/platform identity difficult to predict.
    """
    if pred_logits is None:
        raise ValueError("pred_logits cannot be None when domain adversarial training is enabled.")

    n_classes = int(pred_logits.shape[1])
    if n_classes < 2:
        raise ValueError("Domain adversarial training requires at least two slices/domains.")

    targets = targets.reshape(-1).long().to(pred_logits.device)
    eps = float(label_smoothing)
    if not 0.0 <= eps < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")

    if eps == 0.0:
        return F.cross_entropy(pred_logits, targets, reduction=reduction)

    log_probs = F.log_softmax(pred_logits, dim=1)
    with torch.no_grad():
        smooth_targets = torch.full_like(log_probs, eps / (n_classes - 1))
        smooth_targets.scatter_(1, targets[:, None], 1.0 - eps)

    loss = -(smooth_targets * log_probs).sum(dim=1)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def spatial_contrastive_loss(latent, pos, d_pos, distance=None):
    """Spatial contrastive loss computed within the current spatial block.

    Parameters
    ----------
    latent
        Latent representation for the current block, shape [N, D].
    pos
        Spatial coordinates for the current block, shape [N, 2].
    d_pos
        Radius used to define spatial positive pairs.
    distance
        Optional precomputed pairwise distance matrix for the current block.
        If None, it is computed on demand with torch.cdist.
    """
    latent = F.normalize(latent, dim=1)
    sim = torch.matmul(latent, latent.T)
    eye_bool = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye_bool, float("-inf"))

    if distance is None:
        distance = torch.cdist(pos, pos)

    eye = torch.eye(distance.size(0), device=distance.device)
    pos_mask = ((distance <= d_pos).float() - eye).clamp(min=0)
    neg_mask = (distance > d_pos).float()
    valid = pos_mask.sum(1) > 0
    if not torch.any(valid):
        return torch.zeros((), device=latent.device, dtype=latent.dtype)

    sim_pos = sim.masked_fill(pos_mask == 0, float("-inf"))
    sim_all = sim.masked_fill((pos_mask + neg_mask) == 0, float("-inf"))
    pos_logsumexp = sim_pos.logsumexp(dim=1)[valid]
    all_logsumexp = sim_all.logsumexp(dim=1)[valid]
    return -(pos_logsumexp - all_logsumexp).mean()


def _freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True


def _leiden_pseudo_labels(
    features,
    n_clusters=None,
    resolution=1.0,
    res_range=(0.1, 3.0),
    res_step=0.1,
    n_neighbors=25,
    seed=42,
):
    adata = sc.AnnData(np.asarray(features, dtype=np.float32))
    n_neighbors = min(int(n_neighbors), max(2, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")

    if n_clusters is None:
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            key_added="leiden",
            random_state=seed,
        )
        return adata.obs["leiden"].astype(int).to_numpy()

    best_labels = None
    best_diff = np.inf
    for res in np.arange(res_range[0], res_range[1] + res_step / 2, res_step):
        sc.tl.leiden(
            adata,
            resolution=float(res),
            key_added="leiden_tmp",
            random_state=seed,
        )
        labels = adata.obs["leiden_tmp"].astype(int).to_numpy()
        diff = abs(len(np.unique(labels)) - int(n_clusters))
        if diff < best_diff:
            best_diff = diff
            best_labels = labels.copy()
        if diff == 0:
            break
    return best_labels


def _leiden_subsample_label_transfer(
    features,
    slice_ids=None,

    # sampling
    frac=0.1,
    min_per_slice=2000,
    n_subsample=None,

    # Leiden
    n_neighbors=25,
    n_clusters=None,
    resolution=1.0,
    res_range=(0.1, 3.0),
    res_step=0.1,

    # transfer
    transfer_k=5,
    weighted=True,
    transfer_chunk_size=100000,

    seed=42,
):
    """
    Subsample Leiden clustering + kNN label transfer.

    Parameters
    ----------
    features
        Joint IDEA-N latent representation, shape [N, D].

    slice_ids
        Integer slice IDs aligned with features.

        Multi-section:
            balanced sampling is performed within each slice.

        None:
            global sampling is performed.

    frac
        Sampling fraction for each slice.

    min_per_slice
        Minimum sampled spots per slice.

    n_subsample
        Number of sampled spots for single-section/global mode.
        If None, use min(200000, N).

    n_neighbors
        Number of neighbors for Leiden graph construction.

    n_clusters
        Desired number of Leiden clusters. If not None,
        resolution search is performed.

    resolution
        Fixed Leiden resolution when n_clusters is None.

    transfer_k
        Number of neighbors used for label transfer.

    weighted
        Whether to use inverse-distance weighted voting.

    transfer_chunk_size
        Number of spots transferred at once. This avoids creating
        an N x transfer_k distance/index matrix for very large data.

    seed
        Random seed.

    Returns
    -------
    labels
        Cluster labels for ALL spots, shape [N].

    sub_idx
        Global indices of sampled spots.

    sub_labels
        Leiden labels of sampled spots.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.ndim != 2:
        raise ValueError(
            "features must have shape [N, D]."
        )

    N = int(
        features.shape[0]
    )

    if N < 2:
        raise ValueError(
            "At least two spots are required."
        )

    rng = np.random.default_rng(
        seed
    )

    all_idx = np.arange(
        N,
        dtype=np.int64,
    )

    # ========================================================
    # 1. Subsampling
    # ========================================================

    if slice_ids is not None:

        slice_ids = np.asarray(
            slice_ids
        )

        if len(slice_ids) != N:
            raise ValueError(
                "slice_ids must have the same length "
                "as features."
            )

        sub_idx_list = []

        for slice_id in np.unique(
            slice_ids
        ):

            idx_s = all_idx[
                slice_ids == slice_id
            ]

            n_slice = len(
                idx_s
            )

            n_take = max(
                int(min_per_slice),
                int(
                    n_slice
                    * float(frac)
                ),
            )

            n_take = min(
                n_take,
                n_slice,
            )

            chosen = rng.choice(
                idx_s,
                size=n_take,
                replace=False,
            )

            sub_idx_list.append(
                chosen
            )

            print(
                f"[IDEA-N Leiden] "
                f"slice={slice_id}: "
                f"{n_take:,}/{n_slice:,} sampled"
            )

        sub_idx = np.concatenate(
            sub_idx_list
        )

    else:

        if n_subsample is None:
            n_subsample = min(
                200000,
                N,
            )

        n_subsample = min(
            int(n_subsample),
            N,
        )

        sub_idx = rng.choice(
            N,
            size=n_subsample,
            replace=False,
        )

    sub_idx = np.sort(
        sub_idx
    )

    if len(sub_idx) < 2:
        raise ValueError(
            "Too few sampled spots for Leiden."
        )

    print(
        f"[IDEA-N Leiden] Total sampled: "
        f"{len(sub_idx):,}/{N:,} "
        f"({len(sub_idx) / N:.2%})"
    )

    # ========================================================
    # 2. Subsample latent
    # ========================================================

    X_sub = features[
        sub_idx
    ]

    sub_ad = sc.AnnData(
        X=X_sub
    )

    n_neighbors_eff = min(
        int(n_neighbors),
        len(sub_idx) - 1,
    )

    # ========================================================
    # 3. Leiden graph
    # ========================================================

    sc.pp.neighbors(
        sub_ad,
        n_neighbors=n_neighbors_eff,
        use_rep="X",
        n_pcs=None,
    )

    # ========================================================
    # 4. Leiden clustering
    # ========================================================

    if n_clusters is None:

        sc.tl.leiden(
            sub_ad,
            resolution=float(
                resolution
            ),
            key_added="_leiden_sub",
            random_state=0,
        )

        sub_labels = (
            sub_ad.obs[
                "_leiden_sub"
            ]
            .astype(str)
            .to_numpy()
        )

    else:

        best_labels = None
        best_resolution = None
        best_diff = np.inf

        for res in np.arange(
            res_range[0],
            res_range[1]
            + res_step / 2,
            res_step,
        ):

            sc.tl.leiden(
                sub_ad,
                resolution=float(
                    res
                ),
                key_added="_leiden_tmp",
                random_state=0,
            )

            labels_tmp = (
                sub_ad.obs[
                    "_leiden_tmp"
                ]
                .astype(str)
                .to_numpy()
            )

            n_found = len(
                np.unique(
                    labels_tmp
                )
            )

            diff = abs(
                n_found
                - int(n_clusters)
            )

            if diff < best_diff:

                best_diff = diff

                best_labels = (
                    labels_tmp.copy()
                )

                best_resolution = float(
                    res
                )

            if diff == 0:
                break

        if best_labels is None:
            raise RuntimeError(
                "Leiden resolution search failed."
            )

        sub_labels = best_labels

        print(
            f"[IDEA-N Leiden] "
            f"selected resolution="
            f"{best_resolution:.3f}, "
            f"clusters="
            f"{len(np.unique(sub_labels))}"
        )

    # ========================================================
    # 5. Convert labels -> contiguous integer labels
    # ========================================================

    categorical = pd.Categorical(
        sub_labels
    )

    label_names = np.asarray(
        categorical.categories.astype(
            str
        )
    )

    y_sub = (
        categorical.codes
        .astype(
            np.int32
        )
    )

    n_classes = len(
        label_names
    )

    print(
        f"[IDEA-N Leiden] "
        f"{n_classes} clusters found "
        f"on sampled spots."
    )

    # ========================================================
    # 6. Fit kNN label-transfer model
    # ========================================================

    transfer_k_eff = min(
        int(transfer_k),
        len(sub_idx),
    )

    if transfer_k_eff <= 0:
        raise ValueError(
            "transfer_k must be positive."
        )

    nn_model = NearestNeighbors(
        n_neighbors=transfer_k_eff,
        metric="euclidean",
        n_jobs=-1,
    )

    nn_model.fit(
        X_sub
    )

    # ========================================================
    # 7. Chunked label transfer
    #
    # Important for million-scale data:
    #
    # don't do:
    # dists, inds = nn.kneighbors(features)
    #
    # because this simultaneously creates
    # N x transfer_k arrays.
    # ========================================================

    y_pred = np.empty(
        N,
        dtype=np.int32,
    )

    chunk_size = int(
        transfer_chunk_size
    )

    if chunk_size <= 0:
        raise ValueError(
            "transfer_chunk_size must be positive."
        )

    for start in tqdm(
        range(
            0,
            N,
            chunk_size,
        ),
        desc="[IDEA-N] label transfer",
    ):

        end = min(
            start + chunk_size,
            N,
        )

        X_chunk = features[
            start:end
        ]

        dists, inds = (
            nn_model.kneighbors(
                X_chunk,
                return_distance=True,
            )
        )

        neigh_labels = y_sub[
            inds
        ]

        # ----------------------------------------------------
        # k = 1
        # ----------------------------------------------------

        if transfer_k_eff == 1:

            chunk_pred = (
                neigh_labels[:, 0]
            )

        # ----------------------------------------------------
        # weighted voting
        # ----------------------------------------------------

        elif weighted:

            weights = (
                1.0
                / (
                    dists
                    + 1e-8
                )
            ).astype(
                np.float32,
                copy=False,
            )

            m = end - start

            score = np.zeros(
                (
                    m,
                    n_classes,
                ),
                dtype=np.float32,
            )

            row_idx = np.repeat(
                np.arange(
                    m,
                    dtype=np.int64,
                ),
                transfer_k_eff,
            )

            np.add.at(
                score,
                (
                    row_idx,
                    neigh_labels.reshape(
                        -1
                    ),
                ),
                weights.reshape(
                    -1
                ),
            )

            chunk_pred = (
                score.argmax(
                    axis=1
                )
                .astype(
                    np.int32
                )
            )

            del score
            del weights

        # ----------------------------------------------------
        # majority voting
        # ----------------------------------------------------

        else:

            m = end - start

            votes = np.zeros(
                (
                    m,
                    n_classes,
                ),
                dtype=np.int16,
            )

            row_idx = np.repeat(
                np.arange(
                    m,
                    dtype=np.int64,
                ),
                transfer_k_eff,
            )

            np.add.at(
                votes,
                (
                    row_idx,
                    neigh_labels.reshape(
                        -1
                    ),
                ),
                1,
            )

            chunk_pred = (
                votes.argmax(
                    axis=1
                )
                .astype(
                    np.int32
                )
            )

            del votes

        y_pred[
            start:end
        ] = chunk_pred

        del dists
        del inds
        del neigh_labels
        del X_chunk

    # ========================================================
    # 8. Return integer labels
    #
    # Stage 2 expects integer pseudo_labels.
    # ========================================================

    labels_all = y_pred.astype(
        np.int64
    )

    sub_labels_int = y_sub.astype(
        np.int64
    )

    del sub_ad
    del X_sub

    return (
        labels_all,
        sub_idx,
        sub_labels_int,
    )

def _stratified_sample_indices(
    labels,
    max_per_group=None,
    random_state=42,
):
    """
    Stratified row sampling by niche label.

    Parameters
    ----------
    labels : array-like
        Niche labels for all observations in the current sample.

    max_per_group : int or None
        Maximum number of observations retained for each niche.
        If None, all observations are retained.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Sorted row indices.
    """
    labels = np.asarray(labels)

    n_obs = int(labels.shape[0])

    if max_per_group is None:
        return np.arange(n_obs, dtype=np.int64)

    max_per_group = int(max_per_group)

    if max_per_group <= 0:
        raise ValueError(
            "max_per_group must be a positive integer or None."
        )

    rng = np.random.default_rng(
        int(random_state)
    )

    selected = []

    for label in np.unique(labels):

        idx = np.flatnonzero(
            labels == label
        )

        if len(idx) > max_per_group:

            idx = rng.choice(
                idx,
                size=max_per_group,
                replace=False,
            )

        selected.append(
            np.asarray(
                idx,
                dtype=np.int64,
            )
        )

    if len(selected) == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    selected = np.concatenate(
        selected
    )

    # Sorting improves memmap row-access locality.
    selected.sort()

    return selected




def _run_niche_de(
    X,
    labels,
    gene_names,
    high_genes=None,
    sample_name=None,
    normalize=True,
    target_sum=1e4,
    method="wilcoxon",
):
    """
    Run niche-vs-rest differential expression.

    Parameters
    ----------
    X
        Expression matrix, shape [N, G].

    labels
        IDEA-N niche labels, shape [N].

    gene_names
        Gene names corresponding to X columns.

    high_genes
        Candidate genes to retain in the returned table.

        Important:
            DE itself is still performed over all genes in X,
            preserving the previous behavior.

    sample_name
        Optional section/sample name.

    normalize
        Whether to normalize_total + log1p.

    target_sum
        Library size after normalization.

    method
        Method passed to scanpy.tl.rank_genes_groups.
    """

    gene_names = np.asarray(
        gene_names,
        dtype=str,
    )

    labels = np.asarray(
        labels
    ).astype(str)

    if X.shape[0] != len(
        labels
    ):

        raise ValueError(
            f"X has {X.shape[0]} observations, "
            f"but labels has {len(labels)}."
        )

    if X.shape[1] != len(
        gene_names
    ):

        raise ValueError(
            f"X has {X.shape[1]} genes, "
            f"but gene_names has {len(gene_names)}."
        )

    # ========================================================
    # Temporary AnnData
    # ========================================================

    adata = sc.AnnData(
        X=X,
    )

    adata.var_names = gene_names

    adata.obs["niche"] = pd.Categorical(
        labels
    )

    # ========================================================
    # Normalization
    # ========================================================

    if normalize:

        sc.pp.normalize_total(
            adata,
            target_sum=target_sum,
        )

        sc.pp.log1p(
            adata
        )

    # ========================================================
    # DE
    #
    # One call is sufficient.
    # Each niche is automatically compared against the rest.
    # ========================================================

    sc.tl.rank_genes_groups(
        adata,
        groupby="niche",
        method=method,
        reference="rest",
        pts=True,
    )

    # ========================================================
    # Candidate gene filter
    # ========================================================

    if high_genes is None:

        gene_set = set(
            gene_names
        )

    else:

        gene_set = {
            str(g)
            for g in high_genes
        }

    de_results = []

    groups = (
        adata.obs["niche"]
        .cat.categories
    )

    rg = adata.uns[
        "rank_genes_groups"
    ]

    # ========================================================
    # Extract DE statistics
    # ========================================================

    for group in groups:

        group = str(
            group
        )

        names = np.asarray(
            rg["names"][group]
        )

        pvals = np.asarray(
            rg["pvals"][group]
        )

        pvals_adj = np.asarray(
            rg["pvals_adj"][group]
        )

        logfc = np.asarray(
            rg["logfoldchanges"][group]
        )

        for gene, p, padj, fc in zip(
            names,
            pvals,
            pvals_adj,
            logfc,
        ):

            gene = str(
                gene
            )

            if gene not in gene_set:
                continue

            row = {
                "gene":
                    gene,

                "niche":
                    group,

                "pval":
                    float(p),

                "fdr":
                    float(padj),

                "logFC":
                    float(fc),
            }

            if sample_name is not None:

                row["sample"] = str(
                    sample_name
                )

            de_results.append(
                row
            )

    return pd.DataFrame(
        de_results
    )


class IDEANModel:
    """High-level IDEA-N wrapper with optional conditional reconstruction."""

    def __init__(
        self,
        adatas,
        train_set,
        eval_set,
        gene_names,
        offsets,
        metadata,
        use_condition=False,
        hidden_size=512,
        latent_size=128,
        dropout=0.0,
        var_eps=1e-4,
        max_cluster=100,
        knn_k=25,
        spatial_weight=15.0,
        radius_scale=2.5,
        lr_encoder=1e-3,
        lr_decoder=1e-3,
        lr_classifier=1e-4,
        classifier_weight=10.0,
        use_domain_adversarial=False,
        domain_adv_weight=1.0,
        domain_adv_gamma=10.0,
        domain_label_smoothing=0.1,
        lr_domain=None,
        seed=42,
        use_gpu=None,
        optim = 'NovoGrad'
    ):
        if use_gpu is False:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(seed)
        self.seed = seed
        self.adatas = adatas
        self.train_set = train_set
        self.eval_set = eval_set
        # Backward-compatible alias: st_set now means the non-overlapping eval set.
        self.st_set = eval_set
        self.gene_names = list(gene_names)
        self.offsets = list(offsets)
        self.metadata = dict(metadata)
        self.condition_names = list(self.metadata.get("condition_names", []))
        self.n_conditions = int(self.metadata.get("n_conditions", len(self.condition_names) or 1))
        self.use_condition = bool(use_condition)
        self.use_domain_adversarial = bool(use_domain_adversarial)
        if self.use_domain_adversarial and self.n_conditions < 2:
            raise ValueError(
                "use_domain_adversarial=True requires multi-section data "
                "with at least two slice/domain IDs."
            )

        self.domain_adv_weight = float(domain_adv_weight)
        self.domain_adv_gamma = float(domain_adv_gamma)
        self.domain_label_smoothing = float(domain_label_smoothing)
        self.lr_domain = float(lr_decoder if lr_domain is None else lr_domain)

        self.knn_k = int(knn_k)
        self.spatial_weight = float(spatial_weight)
        self.radius_scale = float(radius_scale)
        self.classifier_weight = float(classifier_weight)
        self.max_cluster = int(max_cluster)
        self.n_clusters = None
        self.pseudo_labels = None

        # subsampled Leiden information
        self.leiden_subsample_idx = None
        self.leiden_subsample_labels = None

        self.history = pd.DataFrame()
        self.optim = optim

        self.config = {
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "dropout": dropout,
            "var_eps": var_eps,
            "max_cluster": max_cluster,
            "knn_k": knn_k,
            "spatial_weight": spatial_weight,
            "radius_scale": radius_scale,
            "lr_encoder": lr_encoder,
            "lr_decoder": lr_decoder,
            "lr_classifier": lr_classifier,
            "classifier_weight": classifier_weight,
            "use_domain_adversarial": self.use_domain_adversarial,
            "domain_adv_weight": self.domain_adv_weight,
            "domain_adv_gamma": self.domain_adv_gamma,
            "domain_label_smoothing": self.domain_label_smoothing,
            "lr_domain": self.lr_domain,
            "seed": seed,
            "use_condition": self.use_condition,
            "n_conditions": self.n_conditions,
            "condition_names": self.condition_names,
        }

        self.model = NicheNetwork(
            input_size=len(self.gene_names),
            hidden_size=hidden_size,
            latent_size=latent_size,
            dropout=dropout,
            var_eps=var_eps,
            max_cluster=max_cluster,
            n_conditions=self.n_conditions,
            use_condition=self.use_condition,
            use_domain_adversarial=self.use_domain_adversarial,
        ).to(self.device)

        import torch_optimizer as optim1

        stage1_param_groups = [
            {
                "params": self.model.vae_encoder.parameters(),
                "lr": lr_encoder,
                "betas": (0.90, 0.98),
                "weight_decay": 1e-2,
            },
            {
                "params": self.model.vae_decoder.parameters(),
                "lr": lr_decoder,
                "betas": (0.90, 0.98),
                "weight_decay": 1e-2,
            },
        ]
        stage1_param_groups = []

        if self.optim == "NovoGrad":

            stage1_param_groups.extend([
                {
                    "params": self.model.vae_encoder.parameters(),
                    "lr": lr_encoder,
                    "betas": (0.90, 0.98),
                    "weight_decay": 1e-2,
                },
                {
                    "params": self.model.vae_decoder.parameters(),
                    "lr": lr_decoder,
                    "betas": (0.90, 0.98),
                    "weight_decay": 1e-2,
                },
            ])

            if self.use_domain_adversarial:
                stage1_param_groups.append(
                    {
                        "params": self.model.batch_classifier.parameters(),
                        "lr": self.lr_domain,
                        "betas": (0.90, 0.98),
                        "weight_decay": 1e-2,
                    }
                )

            self.optimizer_stage1 = optim1.NovoGrad(
                stage1_param_groups
            )


        elif self.optim == "RMSprop":

            stage1_param_groups.extend([
                {
                    "params": self.model.vae_encoder.parameters(),
                    "lr": lr_encoder,
                    "weight_decay": 1e-3,
                    "eps": 0.01,
                    "alpha": 0.85,
                },
                {
                    "params": self.model.vae_decoder.parameters(),
                    "lr": lr_decoder,
                    "weight_decay": 1e-3,
                    "eps": 0.01,
                    "alpha": 0.85,
                },
            ])

            if self.use_domain_adversarial:
                stage1_param_groups.append(
                    {
                        "params": self.model.batch_classifier.parameters(),
                        "lr": self.lr_domain,
                        "weight_decay": 1e-3,
                        "eps": 0.01,
                        "alpha": 0.85,
                    }
                )

            self.optimizer_stage1 = torch.optim.RMSprop(
                stage1_param_groups
            )

        else:
            raise ValueError(
                f"Unsupported optimizer: {self.optim}. "
                "Choose from ['NovoGrad', 'RMSprop']."
            )


        
        self.optimizer_stage2 = optim1.NovoGrad(
            self.model.domain_classifier.parameters(),
            lr=lr_classifier,
            betas=(0.90, 0.98),
            weight_decay=1e-2,
        )

        _freeze(self.model.domain_classifier)
        if self.model.batch_classifier is not None:
            if self.use_domain_adversarial:
                _unfreeze(self.model.batch_classifier)
            else:
                _freeze(self.model.batch_classifier)

    def _condition_one_hot(self, condition_ids):
        if not self.use_condition:
            return None
        condition_ids = condition_ids.reshape(-1).long().to(self.device)
        if condition_ids.numel() == 0:
            raise ValueError("Empty condition tensor.")
        if condition_ids.min() < 0 or condition_ids.max() >= self.n_conditions:
            raise ValueError("Condition IDs are outside the configured condition range.")
        return F.one_hot(condition_ids, num_classes=self.n_conditions).float()

    def _build_batch_geometry(self, pos, need_pairwise_distance=False):
        """Build spatial geometry for exactly one current block.

        The DataLoader uses batch_size=1, so one DataLoader batch corresponds
        to one spatial block. KNN indices, KNN distances, the recommended
        positive radius, and (optionally) the full pairwise distance matrix
        are all computed on-the-fly from spots in this block only.
        """
        if pos.ndim != 2:
            raise ValueError(f"pos must have shape [N, 2], got {tuple(pos.shape)}.")

        n_spots = int(pos.shape[0])
        if n_spots < 2:
            raise ValueError("A spatial block must contain at least two spots.")

        # KNN is built only among spots in the current block.
        pos_np = pos.detach().cpu().numpy()
        k_eff = min(self.knn_k, n_spots - 1)
        nn_model = NearestNeighbors(n_neighbors=k_eff + 1).fit(pos_np)
        knn_dist_np, knn_idx_np = nn_model.kneighbors(pos_np)

        # Remove the first neighbor (the spot itself).
        knn_idx = torch.as_tensor(
            knn_idx_np[:, 1:],
            dtype=torch.long,
            device=self.device,
        )
        knn_dist = torch.as_tensor(
            knn_dist_np[:, 1:],
            dtype=torch.float32,
            device=self.device,
        )

        # Same radius definition as before: median first-NN distance * scale.
        radius = torch.median(knn_dist[:, 0]) * self.radius_scale

        # Only Stage 1 needs the O(N^2) pairwise distance matrix for the
        # spatial contrastive loss. Stage 2 / embedding / prediction do not.
        distance = torch.cdist(pos, pos) if need_pairwise_distance else None

        return knn_idx, knn_dist, radius, distance

    def summary(self):
        return {
            "device": str(self.device),
            "n_samples": self.metadata.get("n_samples"),
            "n_spots": self.metadata.get("n_spots"),
            "n_genes": self.metadata.get("n_genes"),
            "n_train_blocks": self.metadata.get("n_train_blocks"),
            "n_eval_blocks": self.metadata.get("n_eval_blocks"),
            "use_condition": self.use_condition,
            "condition_names": self.condition_names,
            "use_domain_adversarial": self.use_domain_adversarial,
            "domain_adv_weight": self.domain_adv_weight,
            "n_clusters": self.n_clusters,
        }

    def get_latent(self, use_posterior_mean=False):
        """Return a joint latent representation in global spot order."""
        self.model.eval()
        n_total = self.metadata["n_spots"]
        hidden_size = self.config["hidden_size"]
        rep = np.zeros((n_total, hidden_size), dtype=np.float32)
        seen = np.zeros(n_total, dtype=np.int32)

        loader = DataLoader(self.eval_set, batch_size=1, shuffle=False)
        with torch.no_grad():
            for x, pos, global_idx, _condition_id in tqdm(
                loader, desc="[IDEA-N] embedding"
            ):
                x = x.squeeze(0).to(self.device).float()
                pos = pos.squeeze(0).to(self.device).float()
                knn_idx, knn_dist, _, _ = self._build_batch_geometry(
                    pos, need_pairwise_distance=False
                )
                z = self.model.encode(
                    x,
                    knn_idx,
                    knn_dist,
                    use_posterior_mean=use_posterior_mean,
                )
                gi = global_idx.squeeze(0).cpu().numpy()
                rep[gi] += z.cpu().numpy()
                seen[gi] += 1

        if np.any(seen == 0):
            raise RuntimeError("Some spots were never covered by any spatial block.")
        return rep / seen[:, None]

    # Backward-compatible internal name used by the previous refactor.
    def _extract_representation(self, use_posterior_mean=False):
        return self.get_latent(use_posterior_mean=use_posterior_mean)

    def train_model(
        self,
        stage1_epochs=100,
        stage2_epochs=500,

        # ========================================================
        # Leiden
        # ========================================================

        n_clusters=None,
        leiden_resolution=1.0,
        leiden_res_range=(0.1, 3.0),
        leiden_res_step=0.1,
        leiden_neighbors=25,

        # ========================================================
        # Leiden mode
        #
        # "full"
        # "subsample"
        # ========================================================

        leiden_mode="full",

        # subsampling
        leiden_subsample_frac=0.1,
        leiden_min_per_slice=2000,
        leiden_n_subsample=None,

        # label transfer
        leiden_transfer_k=5,
        leiden_transfer_weighted=True,
        leiden_transfer_chunk_size=100000,

        use_posterior_mean_for_clustering=False,

        verbose=True,
    ):
        """Two-stage IDEA-N training.

        Stage 1
            Shared KANA/encoder + (optional) conditional decoder are trained by
            ZINB reconstruction and spatial contrastive regularization. When
            ``use_domain_adversarial=True`` for multi-section data, a slice/domain
            discriminator is additionally trained through a gradient-reversal layer.
        Stage 2
            Joint Leiden pseudo-labels from all sections supervise the niche
            classifier while encoder and decoder are frozen.
        """
        records = []
        # One DataLoader batch == one spatial block.
        # With on-the-fly geometry, block order can be shuffled safely.
        train_loader = DataLoader(self.train_set, batch_size=1, shuffle=True)
        eval_loader = DataLoader(self.eval_set, batch_size=1, shuffle=True)

        # ------------------------- Stage 1 -------------------------
        _unfreeze(self.model.vae_encoder)
        _unfreeze(self.model.vae_decoder)
        _freeze(self.model.domain_classifier)
        if self.model.batch_classifier is not None:
            if self.use_domain_adversarial:
                _unfreeze(self.model.batch_classifier)
            else:
                _freeze(self.model.batch_classifier)

        n_stage1_steps = max(stage1_epochs * len(train_loader), 1)

        for epoch in range(stage1_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_rec = 0.0
            epoch_spatial = 0.0
            epoch_domain = 0.0
            epoch_domain_acc = 0.0
            epoch_alpha = 0.0
            progress = tqdm(
                train_loader,
                desc=f"[IDEA-N stage1] {epoch + 1}/{stage1_epochs}",
                disable=not verbose,
            )

            for step, (x, pos, _global_idx, condition_id) in enumerate(progress):
                x = x.squeeze(0).to(self.device).float()
                pos = pos.squeeze(0).to(self.device).float()
                condition_id = condition_id.squeeze(0).reshape(-1).long()
                condition = self._condition_one_hot(condition_id)

                if self.use_domain_adversarial:
                    # DANN schedule from the reference implementation:
                    # alpha starts near 0 and smoothly approaches 1.
                    global_step = epoch * len(train_loader) + step
                    # p = float(global_step) / float(max(n_stage1_steps - 1, 1))
                    p = float(global_step) / (stage1_epochs * len(train_loader))

                    domain_alpha = (
                        2.0 / (1.0 + np.exp(-self.domain_adv_gamma * p)) - 1.0
                    )
                    domain_target = condition_id.to(self.device)
                else:
                    domain_alpha = 0.0
                    domain_target = None

                # One batch is one block: build neighbors/distances only
                # from the spots contained in this current block.
                knn_idx, knn_dist, radius, distance = self._build_batch_geometry(
                    pos, need_pairwise_distance=True
                )

                self.optimizer_stage1.zero_grad()
                out = self.model(
                    x,
                    knn_idx,
                    knn_dist,
                    condition=condition,
                    use_posterior_mean=False,
                    domain_alpha=domain_alpha,
                )
                latent = out[3]
                px_scale, px_r, px_rate, px_dropout = out[4], out[5], out[6], out[7]
                batch_logits = out[11]

                rec = reconst_loss(px_rate, px_r, px_dropout, px_scale, x).mean()
                spatial = spatial_contrastive_loss(
                    latent,
                    pos,
                    d_pos=radius,
                    distance=distance,
                )

                if self.use_domain_adversarial:
                    domain_loss = domain_cross_entropy_loss(
                        batch_logits,
                        domain_target,
                        label_smoothing=self.domain_label_smoothing,
                    )
                    with torch.no_grad():
                        domain_acc = (
                            batch_logits.argmax(dim=1) == domain_target
                        ).float().mean()
                else:
                    domain_loss = torch.zeros((), device=self.device, dtype=rec.dtype)
                    domain_acc = torch.zeros((), device=self.device, dtype=rec.dtype)

                # Positive domain loss is correct here: GradientReversalFunction
                # reverses this gradient only for the encoder/shared representation.
                loss = (
                    rec
                    + self.spatial_weight * spatial
                    + self.domain_adv_weight * domain_loss
                )
                
                if not torch.isfinite(loss):

                    print("\n" + "=" * 70)
                    print("[IDEA-N] Non-finite loss detected")
                    print(f"epoch = {epoch + 1}")
                    print(f"block = {step}")
                
                    print(
                        "condition_id =",
                        torch.unique(condition_id).detach().cpu().numpy()
                    )
                
                    print(
                        "x:",
                        "min =", x.min().item(),
                        "max =", x.max().item(),
                        "mean =", x.mean().item(),
                        "finite =", torch.isfinite(x).all().item(),
                    )
                
                    print(
                        "knn_dist:",
                        "min =", knn_dist.min().item(),
                        "max =", knn_dist.max().item(),
                        "finite =", torch.isfinite(knn_dist).all().item(),
                    )
                
                    print("rec =", rec.item())
                    print("spatial =", spatial.item())
                    if self.use_domain_adversarial:
                        print("domain_loss =", domain_loss.item())
                        print("domain_acc =", domain_acc.item())
                        print("domain_alpha =", domain_alpha)
                
                    names = [
                        "q_m",
                        "q_v",
                        "latent",
                        "px_scale",
                        "px_r",
                        "px_rate",
                        "px_dropout",
                    ]
                
                    tensors = [
                        out[0],
                        out[1],
                        out[3],
                        out[4],
                        out[5],
                        out[6],
                        out[7],
                    ]
                
                    for name, tensor in zip(names, tensors):
                
                        finite = torch.isfinite(tensor)
                
                        print(
                            f"{name}:",
                            f"finite={finite.all().item()}",
                            f"nan={(torch.isnan(tensor)).sum().item()}",
                            f"inf={(torch.isinf(tensor)).sum().item()}",
                        )
                
                        if finite.any():
                            values = tensor[finite]
                
                            print(
                                f"    min={values.min().item():.4e}, "
                                f"max={values.max().item():.4e}, "
                                f"mean={values.mean().item():.4e}"
                            )
                
                    print("=" * 70)
                
                    raise FloatingPointError(
                        f"Non-finite IDEA-N loss at "
                        f"stage1 epoch {epoch + 1}, block {step}."
                    )

                # if not torch.isfinite(loss):
                #     raise FloatingPointError(
                #         f"Non-finite IDEA-N loss at stage1 epoch {epoch + 1}, block {step}."
                #     )

                loss.backward()
                self.optimizer_stage1.step()

                epoch_loss += float(loss.detach())
                epoch_rec += float(rec.detach())
                epoch_spatial += float(spatial.detach())
                if self.use_domain_adversarial:
                    epoch_domain += float(domain_loss.detach())
                    epoch_domain_acc += float(domain_acc.detach())
                    epoch_alpha += float(domain_alpha)
                    progress.set_postfix(
                        loss=epoch_loss / (step + 1),
                        domain=epoch_domain / (step + 1),
                        domain_acc=epoch_domain_acc / (step + 1),
                        alpha=domain_alpha,
                    )
                else:
                    progress.set_postfix(loss=epoch_loss / (step + 1))

            records.append(
                {
                    "stage": 1,
                    "epoch": epoch + 1,
                    "loss": epoch_loss / len(train_loader),
                    "rec_loss": epoch_rec / len(train_loader),
                    "spatial_loss": epoch_spatial / len(train_loader),
                    "domain_loss": (
                        epoch_domain / len(train_loader)
                        if self.use_domain_adversarial
                        else np.nan
                    ),
                    "domain_acc": (
                        epoch_domain_acc / len(train_loader)
                        if self.use_domain_adversarial
                        else np.nan
                    ),
                    "domain_alpha": (
                        epoch_alpha / len(train_loader)
                        if self.use_domain_adversarial
                        else np.nan
                    ),
                    "class_loss": np.nan,
                }
            )

        # ============================================================
        # Joint representation across all slices/platforms
        # ============================================================

        features = self.get_latent(
            use_posterior_mean=(
                use_posterior_mean_for_clustering
            )
        )

        # ============================================================
        # Leiden mode
        # ============================================================

        leiden_mode = str(
            leiden_mode
        ).lower()

        if leiden_mode not in {
            "full",
            "subsample",
        }:

            raise ValueError(
                "leiden_mode must be "
                "'full' or 'subsample'."
            )


        # ============================================================
        # Mode 1: full Leiden
        # ============================================================

        if leiden_mode == "full":

            print(
                "[IDEA-N] Running full Leiden "
                f"on {features.shape[0]:,} spots..."
            )

            pseudo_labels = (
                _leiden_pseudo_labels(
                    features,
                    n_clusters=n_clusters,
                    resolution=leiden_resolution,
                    res_range=leiden_res_range,
                    res_step=leiden_res_step,
                    n_neighbors=leiden_neighbors,
                    seed=self.seed,
                )
            )

            self.leiden_subsample_idx = None
            self.leiden_subsample_labels = None


        # ============================================================
        # Mode 2: sampled Leiden + label transfer
        # ============================================================

        else:

            print(
                "[IDEA-N] Running subsampled Leiden "
                "+ kNN label transfer..."
            )

            # --------------------------------------------------------
            # Construct per-spot slice IDs
            #
            # Global order:
            #
            # sample 0
            # sample 1
            # ...
            #
            # This is the same order used by self.offsets and
            # self.get_latent().
            # --------------------------------------------------------

            slice_ids = None

            if self.n_conditions > 1:

                slice_ids = np.empty(
                    features.shape[0],
                    dtype=np.int32,
                )

                for sample_idx, sample_meta in enumerate(
                    self.metadata["samples"]
                ):

                    start = int(
                        self.offsets[
                            sample_idx
                        ]
                    )

                    n_sample_spots = int(
                        sample_meta[
                            "n_spots"
                        ]
                    )

                    end = (
                        start
                        + n_sample_spots
                    )

                    slice_ids[
                        start:end
                    ] = sample_idx

            # --------------------------------------------------------
            # Sampled Leiden + transfer
            # --------------------------------------------------------

            (
                pseudo_labels,
                sub_idx,
                sub_labels,
            ) = _leiden_subsample_label_transfer(

                features=features,

                # multi-section:
                # balanced sampling within each slice
                slice_ids=slice_ids,

                # sampling
                frac=leiden_subsample_frac,
                min_per_slice=leiden_min_per_slice,
                n_subsample=leiden_n_subsample,

                # Leiden
                n_neighbors=leiden_neighbors,
                n_clusters=n_clusters,
                resolution=leiden_resolution,
                res_range=leiden_res_range,
                res_step=leiden_res_step,

                # transfer
                transfer_k=leiden_transfer_k,
                weighted=leiden_transfer_weighted,
                transfer_chunk_size=(
                    leiden_transfer_chunk_size
                ),

                seed=self.seed,
            )

            self.leiden_subsample_idx = (
                sub_idx
            )

            self.leiden_subsample_labels = (
                sub_labels
            )

            del slice_ids


        # ============================================================
        # Store pseudo labels
        # ============================================================

        self.pseudo_labels = (
            pseudo_labels.astype(
                np.int64
            )
        )

        self.n_clusters = int(
            len(
                np.unique(
                    self.pseudo_labels
                )
            )
        )

        if self.n_clusters > self.max_cluster:

            raise ValueError(
                f"Pseudo-label clustering produced "
                f"{self.n_clusters} niches, "
                f"exceeding "
                f"max_cluster={self.max_cluster}."
            )

        print(
            f"[IDEA-N] Final pseudo labels: "
            f"{self.n_clusters} niches"
        )

        del features
        self.n_clusters = int(len(np.unique(self.pseudo_labels)))
        if self.n_clusters > self.max_cluster:
            raise ValueError(
                f"Pseudo-label clustering produced {self.n_clusters} niches, "
                f"exceeding max_cluster={self.max_cluster}."
            )

        # ------------------------- Stage 2 -------------------------
        _freeze(self.model.vae_encoder)
        _freeze(self.model.vae_decoder)
        if self.model.batch_classifier is not None:
            _freeze(self.model.batch_classifier)
        _unfreeze(self.model.domain_classifier)

        for epoch in range(stage2_epochs):
            self.model.train()
            epoch_class = 0.0
            progress = tqdm(
                eval_loader,
                desc=f"[IDEA-N stage2] {epoch + 1}/{stage2_epochs}",
                disable=not verbose,
            )

            for step, (x, pos, global_idx, _condition_id) in enumerate(progress):
                x = x.squeeze(0).to(self.device).float()
                pos = pos.squeeze(0).to(self.device).float()
                gi = global_idx.squeeze(0).cpu().numpy()
                target = torch.as_tensor(
                    self.pseudo_labels[gi],
                    dtype=torch.long,
                    device=self.device,
                )
                knn_idx, knn_dist, _, _ = self._build_batch_geometry(
                    pos, need_pairwise_distance=False
                )

                self.optimizer_stage2.zero_grad()
                with torch.no_grad():
                    # The niche classifier uses the shared representation only;
                    # condition is not part of this prediction path.
                    z = self.model.encode(
                        x,
                        knn_idx,
                        knn_dist,
                        use_posterior_mean=False,
                    )
                logits = self.model.domain_classifier(z)[:, : self.n_clusters]
                class_loss = F.cross_entropy(logits, target)
                loss = self.classifier_weight * class_loss
                loss.backward()
                self.optimizer_stage2.step()

                epoch_class += float(class_loss.detach())
                progress.set_postfix(class_loss=epoch_class / (step + 1))

            records.append(
                {
                    "stage": 2,
                    "epoch": epoch + 1,
                    "loss": self.classifier_weight * epoch_class / len(eval_loader),
                    "rec_loss": np.nan,
                    "spatial_loss": np.nan,
                    "domain_loss": np.nan,
                    "domain_acc": np.nan,
                    "domain_alpha": np.nan,
                    "class_loss": epoch_class / len(eval_loader),
                }
            )

        self.history = pd.DataFrame(records)
        return self.history

    @torch.no_grad()
    def predict_classifier(self, use_posterior_mean=False, obs_key="niche_pre"):
        """Predict shared niches for all sections and write results back to AnnData."""
        if self.n_clusters is None:
            raise RuntimeError("train_model() must be called before predict().")

        self.model.eval()
        n_total = self.metadata["n_spots"]
        prob_sum = np.zeros((n_total, self.n_clusters), dtype=np.float32)
        rep_sum = np.zeros((n_total, self.config["hidden_size"]), dtype=np.float32)
        seen = np.zeros(n_total, dtype=np.int32)

        loader = DataLoader(self.eval_set, batch_size=1, shuffle=False)
        for x, pos, global_idx, _condition_id in tqdm(
            loader, desc="[IDEA-N] predict"
        ):
            x = x.squeeze(0).to(self.device).float()
            pos = pos.squeeze(0).to(self.device).float()
            knn_idx, knn_dist, _, _ = self._build_batch_geometry(
                pos, need_pairwise_distance=False
            )
            z = self.model.encode(
                x,
                knn_idx,
                knn_dist,
                use_posterior_mean=use_posterior_mean,
            )
            logits = self.model.domain_classifier(z)[:, : self.n_clusters]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            z_np = z.cpu().numpy()
            gi = global_idx.squeeze(0).cpu().numpy()
            prob_sum[gi] += probs
            rep_sum[gi] += z_np
            seen[gi] += 1

        if np.any(seen == 0):
            raise RuntimeError("Some spots received no prediction.")
        probs = prob_sum / seen[:, None]
        reps = rep_sum / seen[:, None]
        labels = probs.argmax(axis=1).astype(str)

        rows = []
        start = 0
        for sample_name, adata in zip(self.metadata["sample_names"], self.adatas):
            end = start + adata.n_obs
            adata.obs[obs_key] = pd.Categorical(labels[start:end])
            adata.obsm["X_IDEA_N"] = reps[start:end]
            adata.obsm["IDEA_N_prob"] = probs[start:end]

            df = pd.DataFrame(
                probs[start:end],
                index=adata.obs_names,
                columns=[f"niche_{i}" for i in range(self.n_clusters)],
            )
            df.insert(0, "niche", labels[start:end])
            df.insert(0, "sample", sample_name)
            rows.append(df)
            start = end

        return pd.concat(rows, axis=0)
    
    def predict(
            self,
            obs_key="niche",
            save_latent=True,
        ):
        
        if self.pseudo_labels is None:
            raise RuntimeError(
                "train_model() must be called before predict()."
            )
    
        labels = self.pseudo_labels.astype(str)
    
        latent = None
    
        if save_latent:
            latent = self.get_latent(
                use_posterior_mean=False
            )
    
        rows = []
    
        start = 0
    
        for sample_name, adata in zip(
            self.metadata["sample_names"],
            self.adatas,
        ):
    
            end = start + adata.n_obs
    
            sample_labels = labels[start:end]
    
            adata.obs[obs_key] = pd.Categorical(
                sample_labels
            )
    
            if latent is not None:
                adata.obsm["X_IDEA_N"] = (
                    latent[start:end]
                )
    
            df = pd.DataFrame(
                {
                    "sample": sample_name,
                    "niche": sample_labels,
                },
                index=adata.obs_names,
            )
    
            rows.append(df)
    
            start = end
    
        return pd.concat(
            rows,
            axis=0
        )
    
    def explain(
        self,
        top_n=30,
        n_steps=50,
        output_dir=None,
        use_posterior_mean=False,
        score_reduction="sum",
        attribution_scope="all",
        normalize_by_target_spots=False,
    ):
    
        from .interpretation import explain_niches
    
        return explain_niches(
            self,
            top_n=top_n,
            n_steps=n_steps,
            output_dir=output_dir,
            use_posterior_mean=use_posterior_mean,
            score_reduction=score_reduction,
            attribution_scope=attribution_scope,
            normalize_by_target_spots=normalize_by_target_spots,
        )
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config,
                "metadata": self.metadata,
                "gene_names": self.gene_names,
                "n_clusters": self.n_clusters,
                "pseudo_labels": self.pseudo_labels,
                "history": self.history.to_dict(orient="list") if not self.history.empty else None,
            },
            path,
        )

    def load(self, path, strict=True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"], strict=strict)
        self.n_clusters = ckpt.get("n_clusters")
        self.pseudo_labels = ckpt.get("pseudo_labels")
        history = ckpt.get("history")
        if history is not None:
            self.history = pd.DataFrame(history)
        return self




    def calculate_de(
        self,
        high_genes=None,
        mode="joint",
        normalize=True,
        target_sum=1e4,
        method="wilcoxon",
        best_only=False,
        max_per_niche=None,
        random_state=None,
        output_path=None,
    ):
        """
        Calculate niche-associated differential expression.

        Parameters
        ----------
        high_genes : list or None
            Candidate genes to retain in the returned DE table.

            None:
                return DE statistics for all genes.

            Important:
                DE itself is still performed over all genes in the
                IDEA-N gene space, preserving the previous behavior.

        mode : {"joint", "per_sample"}
            joint:
                Pool sections and calculate shared-niche DE.

                When max_per_niche is not None, sampling is performed
                independently within each sample and niche before
                pooling. This prevents very large sections from
                dominating the joint sampled dataset.

            per_sample:
                Calculate niche DE independently within each section.

            For a single-section model, both modes are equivalent.

        normalize : bool
            Whether to perform normalize_total + log1p.

        target_sum : float
            Target library size.

        method : str
            Scanpy DE method, e.g. "wilcoxon".

        best_only : bool
            If True, retain only the most significant niche for
            each gene.

        max_per_niche : int or None
            Maximum number of spots retained from each niche.

            None:
                Use all spots. This reproduces the previous full-data
                DE behavior.

            int:
                Use stratified sampling.

                Single section:
                    at most max_per_niche spots per niche.

                joint multi-section:
                    at most max_per_niche spots per
                    sample × niche combination.

                per_sample:
                    at most max_per_niche spots per niche within
                    each sample.

        random_state : int or None
            Sampling seed.

            If None, use self.seed when available, otherwise 42.

        output_path : str or Path or None
            Optional CSV output path.

        Returns
        -------
        pd.DataFrame
        """

        # ========================================================
        # 0. Check state
        # ========================================================

        if self.pseudo_labels is None:

            raise RuntimeError(
                "train_model() must be called before "
                "calculate_de()."
            )

        mode = str(
            mode
        ).lower()

        if mode not in {
            "joint",
            "per_sample",
        }:

            raise ValueError(
                "mode must be 'joint' or 'per_sample'."
            )

        if max_per_niche is not None:

            max_per_niche = int(
                max_per_niche
            )

            if max_per_niche <= 0:

                raise ValueError(
                    "max_per_niche must be a positive "
                    "integer or None."
                )

        if random_state is None:

            random_state = int(
                getattr(
                    self,
                    "seed",
                    42,
                )
            )

        else:

            random_state = int(
                random_state
            )

        # ========================================================
        # 1. Candidate genes
        # ========================================================

        if high_genes is not None:

            high_genes = [
                str(g)
                for g in high_genes
                if str(g) in self.gene_names
            ]

            if len(high_genes) == 0:

                raise ValueError(
                    "None of high_genes are present in "
                    "the IDEA-N gene space."
                )

        # ========================================================
        # 2. Expression matrices
        #
        # IMPORTANT:
        # Use the memmaps instead of model.adatas[i].X.
        #
        # This also works when streaming preparation keeps only
        # lightweight AnnData objects.
        # ========================================================

        if not hasattr(
            self.eval_set,
            "X_mms",
        ):

            raise AttributeError(
                "eval_set does not expose X_mms. "
                "The current calculate_de() implementation "
                "expects the IDEA-N memmap dataset."
            )

        n_samples = len(
            self.eval_set.X_mms
        )

        if n_samples != len(
            self.condition_names
        ):

            raise RuntimeError(
                "Number of expression memmaps does not match "
                "the number of IDEA-N samples."
            )

        # ========================================================
        # 3. Single-section
        # ========================================================

        if n_samples == 1:

            mm = self.eval_set.X_mms[0]

            n_obs = int(
                mm.shape[0]
            )

            labels_all = np.asarray(
                self.pseudo_labels[
                    :n_obs
                ]
            )

            sample_idx = (
                _stratified_sample_indices(
                    labels=labels_all,
                    max_per_group=max_per_niche,
                    random_state=random_state,
                )
            )

            if sample_idx.size == 0:

                raise ValueError(
                    "No observations were selected for DE."
                )

            if max_per_niche is None:

                # Preserve previous full-data behavior.
                X = np.asarray(
                    mm,
                    dtype=np.float32,
                ).copy()

                labels = labels_all

            else:

                # Important:
                # only sampled rows are materialized from memmap.
                X = np.asarray(
                    mm[sample_idx],
                    dtype=np.float32,
                )

                labels = labels_all[
                    sample_idx
                ]

                print(
                    f"[IDEA-N DE] Sampling: "
                    f"{len(sample_idx):,} / {n_obs:,} spots "
                    f"(max {max_per_niche:,} per niche)"
                )

            de_df = _run_niche_de(
                X=X,
                labels=labels,
                gene_names=self.gene_names,
                high_genes=high_genes,
                sample_name=self.condition_names[0],
                normalize=normalize,
                target_sum=target_sum,
                method=method,
            )

            del X
            del labels

        # ========================================================
        # 4. Multi-section: joint DE
        # ========================================================

        elif mode == "joint":

            # ----------------------------------------------------
            # Full-data mode:
            #
            # Preserve the previous behavior exactly.
            # ----------------------------------------------------

            if max_per_niche is None:

                matrices = []

                for mm in self.eval_set.X_mms:

                    matrices.append(
                        np.asarray(
                            mm,
                            dtype=np.float32,
                        )
                    )

                X = np.concatenate(
                    matrices,
                    axis=0,
                )

                labels = self.pseudo_labels[
                    :X.shape[0]
                ]

                del matrices

            # ----------------------------------------------------
            # Sampled joint mode:
            #
            # Sample each sample × niche independently first.
            #
            # This avoids loading complete memmaps and avoids the
            # largest section dominating the pooled sampled data.
            # ----------------------------------------------------

            else:

                sampled_X = []
                sampled_labels = []

                total_before = 0
                total_after = 0

                for sample_idx, (
                    sample_name,
                    mm,
                ) in enumerate(
                    zip(
                        self.condition_names,
                        self.eval_set.X_mms,
                    )
                ):

                    n_obs = int(
                        mm.shape[0]
                    )

                    start = int(
                        self.offsets[
                            sample_idx
                        ]
                    )

                    end = (
                        start
                        + n_obs
                    )

                    labels_all = np.asarray(
                        self.pseudo_labels[
                            start:end
                        ]
                    )

                    row_idx = (
                        _stratified_sample_indices(
                            labels=labels_all,
                            max_per_group=max_per_niche,
                            random_state=(
                                random_state
                                + sample_idx
                            ),
                        )
                    )

                    if row_idx.size == 0:
                        continue

                    X_sample = np.asarray(
                        mm[row_idx],
                        dtype=np.float32,
                    )

                    labels_sample = (
                        labels_all[
                            row_idx
                        ]
                    )

                    sampled_X.append(
                        X_sample
                    )

                    sampled_labels.append(
                        labels_sample
                    )

                    total_before += n_obs
                    total_after += len(
                        row_idx
                    )

                    print(
                        f"[IDEA-N DE] {sample_name}: "
                        f"{len(row_idx):,} / {n_obs:,} spots "
                        f"(max {max_per_niche:,} per niche)"
                    )

                if len(sampled_X) == 0:

                    raise ValueError(
                        "No observations were selected for joint DE."
                    )

                X = np.concatenate(
                    sampled_X,
                    axis=0,
                )

                labels = np.concatenate(
                    sampled_labels,
                    axis=0,
                )

                print(
                    f"[IDEA-N DE] Joint sampled dataset: "
                    f"{total_after:,} / {total_before:,} spots"
                )

                del sampled_X
                del sampled_labels

            de_df = _run_niche_de(
                X=X,
                labels=labels,
                gene_names=self.gene_names,
                high_genes=high_genes,
                sample_name=None,
                normalize=normalize,
                target_sum=target_sum,
                method=method,
            )

            # Mark pooled analysis
            de_df.insert(
                0,
                "sample",
                "joint",
            )

            del X
            del labels

        # ========================================================
        # 5. Multi-section: per-sample DE
        # ========================================================

        else:

            result_list = []

            for sample_idx, (
                sample_name,
                mm,
            ) in enumerate(
                zip(
                    self.condition_names,
                    self.eval_set.X_mms,
                )
            ):

                print(
                    f"[IDEA-N DE] "
                    f"{sample_name}"
                )

                n_obs = int(
                    mm.shape[0]
                )

                # ----------------------------------------------
                # Global pseudo-label interval
                # ----------------------------------------------

                start = int(
                    self.offsets[
                        sample_idx
                    ]
                )

                end = (
                    start
                    + n_obs
                )

                labels_all = np.asarray(
                    self.pseudo_labels[
                        start:end
                    ]
                )

                row_idx = (
                    _stratified_sample_indices(
                        labels=labels_all,
                        max_per_group=max_per_niche,
                        random_state=(
                            random_state
                            + sample_idx
                        ),
                    )
                )

                if row_idx.size == 0:

                    print(
                        f"[IDEA-N DE] Skip {sample_name}: "
                        f"no sampled observations."
                    )

                    continue

                if max_per_niche is None:

                    X = np.asarray(
                        mm,
                        dtype=np.float32,
                    ).copy()

                    labels = labels_all

                else:

                    X = np.asarray(
                        mm[row_idx],
                        dtype=np.float32,
                    )

                    labels = labels_all[
                        row_idx
                    ]

                    print(
                        f"[IDEA-N DE] Sampling: "
                        f"{len(row_idx):,} / {n_obs:,} spots "
                        f"(max {max_per_niche:,} per niche)"
                    )

                current_df = (
                    _run_niche_de(
                        X=X,
                        labels=labels,
                        gene_names=self.gene_names,
                        high_genes=high_genes,
                        sample_name=sample_name,
                        normalize=normalize,
                        target_sum=target_sum,
                        method=method,
                    )
                )

                result_list.append(
                    current_df
                )

                del X
                del labels

            if len(result_list) == 0:

                raise ValueError(
                    "No per-sample DE results were generated."
                )

            de_df = pd.concat(
                result_list,
                axis=0,
                ignore_index=True,
            )

        # ========================================================
        # 6. Best niche for each candidate gene
        # ========================================================

        if best_only:

            if (
                mode == "per_sample"
                and n_samples > 1
            ):

                de_df = (
                    de_df
                    .sort_values(
                        [
                            "sample",
                            "gene",
                            "fdr",
                            "logFC",
                        ],
                        ascending=[
                            True,
                            True,
                            True,
                            False,
                        ],
                    )
                    .groupby(
                        [
                            "sample",
                            "gene",
                        ],
                        as_index=False,
                    )
                    .first()
                )

            else:

                de_df = (
                    de_df
                    .sort_values(
                        [
                            "gene",
                            "fdr",
                            "logFC",
                        ],
                        ascending=[
                            True,
                            True,
                            False,
                        ],
                    )
                    .groupby(
                        "gene",
                        as_index=False,
                    )
                    .first()
                )

        # ========================================================
        # 7. Sort final table
        # ========================================================

        sort_cols = [
            col
            for col in [
                "sample",
                "gene",
                "fdr",
                "logFC",
            ]
            if col in de_df.columns
        ]

        de_df = (
            de_df
            .sort_values(
                sort_cols,
                ascending=[
                    True
                    if col != "logFC"
                    else False
                    for col in sort_cols
                ],
            )
            .reset_index(
                drop=True
            )
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
                exist_ok=True,
            )

            de_df.to_csv(
                output_path,
                index=False,
            )

        return de_df