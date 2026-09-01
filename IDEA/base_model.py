from __future__ import annotations

import random
from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scvi.distributions import ZeroInflatedNegativeBinomial
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm


EPS = 0.001


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    index = index.reshape(-1, 1).long()
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index, 1)
    return onehot.float()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale_dim, dropout=0.0):
        super().__init__()
        self.scale_dim = scale_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # Kept identical to the original code: scale by sqrt(d_model), not sqrt(d_k).
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.scale_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return context, attn


class MultiHeadAttention(nn.Module):
    """Retained for checkpoint compatibility; currently not used in Decoder.forward."""

    def __init__(self, d_model=512, d_k=64, d_v=64, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        residual = input_Q
        batch_size = input_Q.size(0)
        batch_size_k = input_K.size(0)
        batch_size_v = input_V.size(0)
        q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(input_K).view(batch_size_k, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(input_V).view(batch_size_v, -1, self.n_heads, self.d_v).transpose(1, 2)
        context, attn = ScaledDotProductAttention(self.d_model)(q, k, v)
        context = context.transpose(1, 2).reshape(batch_size_k, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output + residual, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.norm = nn.LayerNorm(d_model, eps=EPS)

    def forward(self, inputs):
        return self.fc(inputs) + inputs


def precompute_freqs_cis_2d(dim, end, theta=10000.0, use_cls=False, x_pos=None, y_pos=None):
    device = x_pos.device
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: dim // 4].float() / dim)
    )
    x_freqs = torch.outer(x_pos.reshape(-1), freqs).float()
    y_freqs = torch.outer(y_pos.reshape(-1), freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat(
        [x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1
    )
    return freqs_cis.reshape(end if not use_cls else end + 1, -1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError(
            f"RoPE frequency shape {freqs_cis.shape} is incompatible with tensor shape {x.shape}."
        )
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xv_ = torch.view_as_complex(xv.float().reshape(*xv.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    xv_out = torch.view_as_real(xv_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk), xv_out.type_as(xv)


class RAttention(nn.Module):
    def __init__(
        self,
        dim=512,
        num_heads=8,
        d_k=64,
        d_v=64,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Retained because these parameters existed in the original module/state_dict.
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.W_Q = nn.Linear(dim, d_k * num_heads, bias=False)
        self.W_K = nn.Linear(dim, d_k * num_heads, bias=False)
        self.W_V = nn.Linear(dim, d_v * num_heads, bias=False)
        self.fc = nn.Linear(num_heads * d_v, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=0.005)

    def forward(self, q, k, v, x_pos, y_pos):
        B, N, _ = q.shape
        residual = q

        # Recompute from current coordinates. The original cache only checked N,
        # so equally sized blocks could accidentally reuse coordinates from a prior block.
        freqs_cis = precompute_freqs_cis_2d(
            self.dim // self.num_heads,
            N,
            x_pos=x_pos,
            y_pos=y_pos,
        )

        q = self.W_Q(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_K(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_V(v).view(B, -1, self.num_heads, self.d_v).transpose(1, 2)

        q, k, _ = apply_rotary_emb(q, k, v, freqs_cis=freqs_cis)
        context, attn = ScaledDotProductAttention(self.dim)(q, k, v)
        context = context.transpose(1, 2).reshape(B, -1, self.num_heads * self.d_v)
        output = self.fc(context)
        return (output + residual).squeeze(0), attn


class FCResBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0, activation="gelu", norm="layer"):
        super().__init__()
        hidden_dim = hidden_dim or dim

        if norm == "layer":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        elif norm == "batch":
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError("norm must be 'layer' or 'batch'")

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError("Unsupported activation")

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return identity + out


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, label_size, dropout, var_eps):
        super().__init__()
        self.vae_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.FCResBlock_1 = FCResBlock(hidden_size, latent_size)
        self.FCResBlock_2 = FCResBlock(hidden_size, latent_size)
        self.FCResBlock_3 = FCResBlock(hidden_size, latent_size)
        self.mean_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.var_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.var_activation = torch.exp
        self.var_eps = var_eps

    def forward(self, x):
        x = self.vae_encoder(x)
        x = self.FCResBlock_1(x)
        x = self.FCResBlock_2(x)
        q = self.FCResBlock_3(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()
        return q_m, q_v, dist, latent


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, label_size, dropout, var_eps):
        super().__init__()
        self.hidden_size = hidden_size
        self.vae_decoder_b1 = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=EPS),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=EPS),
        )
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1),
        )
        self.FCResBlock_1 = FCResBlock(hidden_size, latent_size)
        self.FCResBlock_2 = FCResBlock(hidden_size, latent_size)
        self.FCResBlock_3 = FCResBlock(hidden_size, latent_size)
        self.px_r_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.px_dropout_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.px_r = nn.Parameter(torch.randn(output_size, 2))

        self.muti_attn = MultiHeadAttention(hidden_size, 64, 64, 8)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_size, hidden_size)
        self.RAttention = RAttention(hidden_size, 8, 64, 64)
        self.pos_ffn1 = PoswiseFeedForwardNet(hidden_size, hidden_size)
        self.RAttention1 = RAttention(hidden_size, 8, 64, 64)
        self.pos_ffn2 = PoswiseFeedForwardNet(hidden_size, hidden_size)
        self.RAttention2 = RAttention(hidden_size, 8, 64, 64)

    def _spatial_attention(self, z, x_pos, y_pos):
        z = z.view(1, -1, self.hidden_size)
        z, _ = self.RAttention(z, z, z, x_pos, y_pos)
        z = self.pos_ffn(z)
        z = z.view(1, -1, self.hidden_size)
        z, _ = self.RAttention1(z, z, z, x_pos, y_pos)
        z = self.pos_ffn1(z)
        z = z.view(1, -1, self.hidden_size)
        z, _ = self.RAttention2(z, z, z, x_pos, y_pos)
        z = self.pos_ffn2(z)
        return z

    def forward(self, z, batch, library, x_pos, y_pos):
        z = self.FCResBlock_1(z)
        z = self.FCResBlock_2(z)

        n_sm = int(batch.sum().item())
        if 0 < n_sm < z.shape[0]:
            z_st = self._spatial_attention(z[n_sm:], x_pos[n_sm:], y_pos[n_sm:])
            z = torch.cat([z[:n_sm], z_st], dim=0)
        elif n_sm == 0:
            z = self._spatial_attention(z, x_pos, y_pos)

        batch_1 = one_hot(batch, 2)
        z1 = torch.cat([z, batch_1], dim=1)
        z1 = self.vae_decoder_b1(z1)
        z1 = torch.cat([z1, batch_1], dim=1)
        z1 = self.vae_decoder_b1(z1)

        px = self.FCResBlock_3(z + z1)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = torch.exp(F.linear(batch_1, self.px_r))
        return px_scale, px_r, px_rate, px_dropout


class DeconvHead(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, label_size, dropout, var_eps):
        super().__init__()
        self.vae_deconv_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=EPS),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size, eps=EPS),
        )
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, label_size)
        self.linear3 = nn.Linear(label_size, label_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)
        self.FCResBlock_1 = FCResBlock(hidden_size, latent_size)
        self.FCResBlock_2 = FCResBlock(hidden_size, latent_size)
        self.FCResBlock_3 = FCResBlock(hidden_size, latent_size)

    def forward(self, z, batch, library, x_pos, y_pos):
        z = self.FCResBlock_1(z)
        z = self.FCResBlock_2(z)
        z_1 = self.vae_deconv_1(z)
        z_1 = self.vae_deconv_1(z_1)
        px = self.FCResBlock_3(z + z_1)
        label_1 = self.linear1(px)
        label_2 = self.linear2(self.relu(label_1))
        label_3 = self.linear3(self.relu(label_2))
        return self.softmax(label_3)


class STSM(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, label_size, dropout, var_eps):
        super().__init__()
        self.vae_encoder = Encoder(input_size, hidden_size, latent_size, output_size, label_size, dropout, var_eps)
        self.vae_decoder = Decoder(input_size, hidden_size, latent_size, output_size, label_size, dropout, var_eps)
        self.deconv = DeconvHead(input_size, hidden_size, latent_size, output_size, label_size, dropout, var_eps)

    def forward(self, x, batch, x_pos, y_pos):
        x_log = torch.log1p(x).float()
        library = torch.log(x.sum(1)).unsqueeze(1).to(torch.float32)
        q_m, q_v, dist, latent = self.vae_encoder(x_log)
        px_scale, px_r, px_rate, px_dropout = self.vae_decoder(
            latent, batch, library, x_pos, y_pos
        )
        label = self.deconv(latent, batch, library, x_pos, y_pos)
        return q_m, q_v, dist, latent, px_scale, px_r, px_rate, px_dropout, label


def reconst_loss(px_rate, px_r, px_dropout, px_scale, x, batch=None):
    px = ZeroInflatedNegativeBinomial(
        mu=px_rate,
        theta=px_r,
        zi_logits=px_dropout,
        scale=px_scale,
    )
    return -px.log_prob(x).sum(-1)


def masked_mse_loss(targets, preds, mask):
    loss = F.mse_loss(preds, targets, reduction="none")
    mask = mask.reshape(-1, 1).to(loss.dtype)
    denom = mask.sum() * preds.shape[-1]
    if denom.item() == 0:
        return torch.zeros((), device=preds.device, dtype=preds.dtype)
    return (loss * mask).sum() / denom


def compute_kernel(x, y, bandwidth=None):
    x_size, y_size, dim = x.size(0), y.size(0), x.size(1)
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
    return torch.exp(-torch.square(tiled_x - tiled_y).mean(dim=2) / dim)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return x_kernel.sum() + y_kernel.sum() - 2 * xy_kernel.sum()


def kl_divergence(y_true, y_pred, dim=0, mask=None):
    if mask is None:
        mask = torch.ones((y_true.shape[0], 1), device=y_true.device, dtype=y_true.dtype)
    mask = mask.reshape(-1, 1).to(y_pred.dtype)
    eps = torch.finfo(torch.float32).eps
    y_pred = torch.clip(y_pred, eps) * mask
    y_true = y_true.to(y_pred.dtype) * mask
    y_true = torch.nan_to_num(y_true / y_true.sum(dim, keepdim=True), 0)
    y_pred = torch.nan_to_num(y_pred / y_pred.sum(dim, keepdim=True), 0)
    y_true = torch.clip(y_true, eps, 1)
    y_pred = torch.clip(y_pred, eps, 1)
    return torch.mul(y_true, torch.log(torch.nan_to_num(y_true / y_pred))).mean(dim)


class IDEACModel:
    """SPACEL-style high-level wrapper for IDEA-C."""

    def __init__(
        self,
        st_ad,
        sc_ad,
        sm_ad,
        st_set,
        sm_set,
        celltype_names,
        hidden_size=512,
        latent_size=128,
        dropout=0.7,
        var_eps=1e-4,
        batch_size=2048,
        learning_rate=1e-3,
        lkl=1000,
        lmmd=1000,
        lmse=1000,
        seed=42,
        use_gpu=None,
        metadata=None,
    ):
        if hidden_size != 512:
            raise ValueError(
                "The current 2D RoPE configuration uses 8 heads × 64 dimensions and therefore requires hidden_size=512."
            )

        if use_gpu is False:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(seed)
        self.seed = seed
        self.st_ad = st_ad
        self.sc_ad = sc_ad
        self.sm_ad = sm_ad
        self.st_set = st_set
        self.sm_set = sm_set
        self.celltype_names = list(celltype_names)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lkl = lkl
        self.lmmd = lmmd
        self.lmse = lmse
        self.metadata = metadata or {}

        self.model = STSM(
            input_size=st_ad.n_vars,
            hidden_size=hidden_size,
            latent_size=latent_size,
            output_size=st_ad.n_vars,
            label_size=len(self.celltype_names),
            dropout=dropout,
            var_eps=var_eps,
        ).to(self.device)

        import torch_optimizer as optim1

        self.optimizer = optim1.NovoGrad(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=1e-2,
        )
        self.history = pd.DataFrame()

    @staticmethod
    def _next_batch(iterator, loader):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator

    def _combine_batches(self, sm_batch, st_batch):
        sm_x, sm_label = sm_batch
        st_x, st_pos, _spot_idx = st_batch
        st_x = st_x.squeeze(0)
        st_pos = st_pos.squeeze(0)

        n_sm = sm_x.shape[0]
        n_st = st_x.shape[0]
        st_label = torch.zeros((n_st, len(self.celltype_names)), dtype=sm_label.dtype)

        label = torch.cat([sm_label, st_label], dim=0)
        x = torch.cat([sm_x, st_x], dim=0)
        batch = torch.cat(
            [torch.ones((n_sm, 1)), torch.zeros((n_st, 1))], dim=0
        )
        pos = torch.cat([torch.zeros((n_sm, 2)), st_pos], dim=0)
        return label, x, batch, pos

    def train_model(self, max_epochs=100, verbose=True):
        sm_loader = DataLoader(self.sm_set, batch_size=self.batch_size, shuffle=True)
        st_loader = DataLoader(self.st_set, batch_size=1, shuffle=False)
        total_batches = max(len(sm_loader), len(st_loader))

        sm_iter = iter(sm_loader)
        st_iter = iter(st_loader)
        records = []
        kl_infer_loss_func = partial(kl_divergence, dim=1)

        for epoch in range(max_epochs):
            self.model.train()
            progress = tqdm(
                range(total_batches),
                desc=f"[train] epoch {epoch + 1}/{max_epochs}",
                disable=not verbose,
            )

            epoch_loss = 0.0
            epoch_rec = 0.0
            epoch_mse = 0.0
            epoch_kl = 0.0
            epoch_mmd = 0.0

            for _ in progress:
                sm_batch, sm_iter = self._next_batch(sm_iter, sm_loader)
                st_batch, st_iter = self._next_batch(st_iter, st_loader)
                label, x, batch, pos = self._combine_batches(sm_batch, st_batch)

                label = label.to(self.device).float()
                label = torch.nan_to_num(
                    label / label.sum(1, keepdim=True), nan=0.0, posinf=0.0, neginf=0.0
                )
                x = x.to(self.device).float()
                batch = batch.to(self.device).float()
                pos = pos.to(self.device).float()
                x_pos = pos[:, 0:1]
                y_pos = pos[:, 1:2]

                self.optimizer.zero_grad()
                (
                    q_m,
                    q_v,
                    dist,
                    latent,
                    px_scale,
                    px_r,
                    px_rate,
                    px_dropout,
                    pred,
                ) = self.model(x, batch, x_pos, y_pos)

                rec = reconst_loss(px_rate, px_r, px_dropout, px_scale, x, batch)
                loss_mse = masked_mse_loss(label, pred, batch)
                loss_kl = kl_infer_loss_func(label.float(), pred, dim=0, mask=batch).sum()

                sm_index = batch.reshape(-1) == 1
                st_index = batch.reshape(-1) == 0
                latent_sm = latent[sm_index]
                latent_st = latent[st_index]

                n = min(latent_sm.size(0), latent_st.size(0))
                if n > 0:
                    if latent_sm.size(0) > n:
                        idx = torch.randperm(latent_sm.size(0), device=self.device)[:n]
                        latent_sm = latent_sm[idx]
                    if latent_st.size(0) > n:
                        idx = torch.randperm(latent_st.size(0), device=self.device)[:n]
                        latent_st = latent_st[idx]

                    # Preserve the original MMD formulation exactly.
                    sm_summary = latent_sm.reshape(n, -1).mean(1).reshape(1, -1)
                    st_summary = latent_st.reshape(n, -1).mean(1).reshape(1, -1)
                    loss_mmd = compute_mmd(sm_summary, st_summary)
                else:
                    loss_mmd = torch.zeros((), device=self.device)

                loss = rec.mean() + self.lkl * loss_kl + self.lmmd * loss_mmd + self.lmse * loss_mse
                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.detach())
                epoch_rec += float(rec.mean().detach())
                epoch_mse += float(loss_mse.detach())
                epoch_kl += float(loss_kl.detach())
                epoch_mmd += float(loss_mmd.detach())

                progress.set_postfix(
                    loss=epoch_loss / max(1, progress.n + 1),
                    deconv_rmse=np.sqrt(epoch_mse / max(1, progress.n + 1)),
                )

            records.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss / total_batches,
                    "rec_loss": epoch_rec / total_batches,
                    "deconv_mse": epoch_mse / total_batches,
                    "kl_loss": epoch_kl / total_batches,
                    "mmd_loss": epoch_mmd / total_batches,
                }
            )

        self.history = pd.DataFrame(records)
        return self.history

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        pred = np.zeros((self.st_ad.n_obs, len(self.celltype_names)), dtype=np.float32)
        loader = DataLoader(self.st_set, batch_size=1, shuffle=False)

        for st_x, st_pos, spot_idx in tqdm(loader, desc="[predict]"):
            st_x = st_x.squeeze(0).to(self.device).float()
            st_pos = st_pos.squeeze(0).to(self.device).float()
            spot_idx_np = spot_idx.squeeze(0).cpu().numpy()
            batch = torch.zeros((st_x.shape[0], 1), device=self.device)
            _, _, _, _, _, _, _, _, block_pred = self.model(
                st_x,
                batch,
                st_pos[:, 0:1],
                st_pos[:, 1:2],
            )
            pred[spot_idx_np] = block_pred.cpu().numpy()

        pred_df = pd.DataFrame(
            pred,
            index=self.st_ad.obs_names,
            columns=self.celltype_names,
        )
        return pred_df

    def save(self, path):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "celltype_names": self.celltype_names,
                "metadata": self.metadata,
            },
            path,
        )

    def load(self, path, strict=True):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=strict)
        return self
    
    def explain(
        self,
        top_n=30,
        n_steps=50,
        output_dir=None,
        use_posterior_mean=True,
    ):
    
        from .interpretation import explain_celltypes
    
        return explain_celltypes(
            self,
            top_n=top_n,
            n_steps=n_steps,
            output_dir=output_dir
        )
