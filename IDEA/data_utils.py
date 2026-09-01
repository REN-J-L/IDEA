from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import anndata as ad

from . import simulation_new
from . import preprocessing
from . import data_downsample_new


class STBlockDataset(Dataset):
    """Spatial dataset that returns one spatial block at a time."""

    def __init__(self, x_memmap, block2spots: Dict[int, np.ndarray], coords: np.ndarray):
        self.X = x_memmap
        self.coords = np.asarray(coords, dtype=np.float32)
        self.block2spots = block2spots
        self.blocks = list(block2spots.keys())

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        block_id = self.blocks[idx]
        spot_idx = self.block2spots[block_id]
        x = torch.from_numpy(np.asarray(self.X[spot_idx], dtype=np.float32))
        coord = torch.from_numpy(self.coords[spot_idx])
        spot_idx = torch.as_tensor(spot_idx, dtype=torch.long)
        return x, coord, spot_idx


class SMDataset(Dataset):
    """Pseudo-spatial dataset containing expression and cell-type proportions."""

    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.as_tensor(self.X[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y[idx], dtype=torch.float32)
        return x, y


def build_blocks(coords, block_size, min_spots=1000):
    """Partition coordinates into grid blocks and merge undersized blocks."""
    coords = np.asarray(coords, dtype=np.float32)
    gx = np.floor(coords[:, 0] / block_size).astype(np.int32)
    gy = np.floor(coords[:, 1] / block_size).astype(np.int32)

    min_gx, min_gy = gx.min(), gy.min()
    gx -= min_gx
    gy -= min_gy

    grid_y = gy.max() + 1
    block_id = gx * grid_y + gy

    block2spots = defaultdict(list)
    for i, b in enumerate(block_id):
        block2spots[b].append(i)

    block_ids = np.array(list(block2spots.keys()), dtype=np.int32)
    block_centers = np.zeros((len(block_ids), 2), dtype=np.float32)
    block_sizes = np.zeros(len(block_ids), dtype=np.int32)

    for i, b in enumerate(block_ids):
        idx = np.asarray(block2spots[b], dtype=np.int32)
        block_centers[i] = coords[idx].mean(axis=0)
        block_sizes[i] = len(idx)

    small_mask = block_sizes < min_spots
    large_mask = ~small_mask

    if small_mask.any() and large_mask.any():
        small_ids = block_ids[small_mask]
        large_ids = block_ids[large_mask]
        small_centers = block_centers[small_mask]
        large_centers = block_centers[large_mask]

        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nn.fit(large_centers)
        _, nearest_idx = nn.kneighbors(small_centers)

        for i, sb in enumerate(small_ids):
            target = large_ids[nearest_idx[i, 0]]
            block2spots[target].extend(block2spots[sb])
            block_id[block_id == sb] = target
            del block2spots[sb]

    new_block2spots = {}
    old2new = {}
    for new_id, b in enumerate(block2spots.keys()):
        new_block2spots[new_id] = np.asarray(block2spots[b], dtype=np.int32)
        old2new[b] = new_id

    new_block_id = np.empty_like(block_id)
    for old_b, new_b in old2new.items():
        new_block_id[block_id == old_b] = new_b

    meta = {
        "min_gx": int(min_gx),
        "min_gy": int(min_gy),
        "block_size": float(block_size),
        "min_spots": int(min_spots),
    }
    return new_block2spots, new_block_id, meta


def estimate_block_size(coords, target_spots_per_block, safety_factor=1.0):
    coords = np.asarray(coords)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    area = (xmax - xmin) * (ymax - ymin)
    if area <= 0:
        raise ValueError("Spatial coordinates must span a non-zero 2D area.")
    density = coords.shape[0] / area
    block_size = np.sqrt(target_spots_per_block / density)
    return block_size * safety_factor


def evaluate_block_size(coords, block_size):
    coords = np.asarray(coords)
    xmin, ymin = coords.min(axis=0)
    block_ids = np.floor((coords - np.array([xmin, ymin])) / block_size).astype(int)

    block2count = defaultdict(int)
    for bid in map(tuple, block_ids):
        block2count[bid] += 1

    counts = np.asarray(list(block2count.values()))
    return {
        "n_blocks": len(counts),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "std": float(counts.std()),
        "min": int(counts.min()),
        "max": int(counts.max()),
    }


def auto_tune_block_size(coords, target, max_spots, tol=0.1, max_iter=10, safety=0.95):
    bs = estimate_block_size(coords, target)
    for _ in range(max_iter):
        stats = evaluate_block_size(coords, bs)
        mean_spots = stats["mean"]
        max_spots_in_block = stats["max"]

        if max_spots_in_block > max_spots:
            bs *= np.sqrt(max_spots / max_spots_in_block) * safety
            continue

        ratio = mean_spots / target
        if abs(ratio - 1) < tol:
            break
        bs /= np.sqrt(ratio)
    return bs


def st_to_memmap(adata, path_x, x_dtype=np.float32, chunk=20000):
    """Write ST expression to a dense memmap in chunks."""
    X = adata.X
    n_spots, n_genes = X.shape

    mmap_x = np.memmap(path_x, dtype=x_dtype, mode="w+", shape=(n_spots, n_genes))
    if hasattr(X, "toarray"):
        for i in range(0, n_spots, chunk):
            mmap_x[i : i + chunk] = X[i : i + chunk].toarray()
    else:
        mmap_x[:] = np.asarray(X)
    mmap_x.flush()
    return mmap_x, n_spots, n_genes


def _to_numpy(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x)

def _load_adata(x):
    if isinstance(x, (str, Path)):
        return ad.read_h5ad(x), True
    return x, False

def prepare_data(
    sc_ad,
    st_ad,
    celltype_key="scelltype",
    spatial_key = 'spatial',
    output_dir="./IDEAC_output",
    target=4096,
    max_spots=6000,
    min_spots=1000,
    n_pseudo=100000,
    n_top_markers=300,
    deg_method="t-test",
    log2fc_min=0.5,
    pval_cutoff=0.01,
    pct_min=0.1,
    only_use_common=True,
    cells_mean=1,
    cells_min=2,
    cells_max=3,
):
    """Prepare real ST and pseudo-spatial reference data for IDEA-C.

    This preserves the preprocessing/simulation sequence used in the original
    ``make_dataset_ct.py`` while exposing it as a reusable library function.
    """

    sc_ad, sc_loaded_here = _load_adata(sc_ad)
    st_ad, st_loaded_here = _load_adata(st_ad)
    


    sc.pp.filter_genes(st_ad, min_cells=1)
    sc.pp.filter_genes(sc_ad, min_cells=1)


    overlapped_genes = np.intersect1d(sc_ad.var_names, st_ad.var_names)
    sc_ad = sc_ad[:, overlapped_genes].copy()
    st_ad = st_ad[:, overlapped_genes].copy()

    sc.pp.filter_cells(st_ad, min_genes=1)
    sc.pp.filter_cells(sc_ad, min_genes=1)

    celltype_names = (
        sc_ad.obs[celltype_key].astype("category").cat.categories.tolist()
    )

    sc_ad, st_ad = preprocessing.cohgenes(
        sc_ad,
        st_ad,
        celltype_key=celltype_key,
        deg_method=deg_method,
        n_top_markers=n_top_markers,
        n_top_hvg=None,
        n_top_hvg_sc=None,
        used_genes=None,
        sc_genes=None,
        st_genes=None,
        log2fc_min=log2fc_min,
        pval_cutoff=pval_cutoff,
        pct_diff=None,
        pct_min=pct_min,
        only_use_common = only_use_common
    )

    sm_ad = simulation_new.generate_sm_adata_new(
        sc_ad,
        celltype_key=celltype_key,
        num_sample=n_pseudo,
        cell_counts=None,
        clusters_mean=None,
        cells_mean=1,
        cells_min=2,
        cells_max=3,
        cell_sample_counts=None,
        cluster_sample_counts=None,
        ncell_sample_list=None,
        cluster_sample_list=None,
        data_augmentation=False,
    )

    data_downsample_new.downsample_sm_spot_counts(sm_ad, st_ad)

    labels = sm_ad.obsm["label"]
    if hasattr(labels, "reindex"):
        labels = labels.reindex(columns=celltype_names, fill_value=0)
    sm_ad.obsm["label"] = labels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    st_ad.write(output_dir / "st_processed.h5ad")
    sm_ad.write(output_dir / "sm_processed.h5ad")

    st_coords = np.asarray(st_ad.obsm[spatial_key], dtype=np.float32)
    block_size = auto_tune_block_size(st_coords, target=target, max_spots=max_spots)
    block2spots, block_id, block_meta = build_blocks(
        st_coords, block_size, min_spots=min_spots
    )

    torch.save(block2spots, output_dir / "st_block2spots.pt")
    np.save(output_dir / "st_coords.npy", st_coords)
    np.save(output_dir / "st_block_id.npy", block_id)

    st_path_x = output_dir / "st_X_memmap.dat"
    st_x_memmap, n_spots, n_genes = st_to_memmap(st_ad, st_path_x)

    sm_x = _to_numpy(sm_ad.uns["down_sample_1"]).astype(np.float32, copy=False)
    sm_label = _to_numpy(sm_ad.obsm["label"]).astype(np.float32, copy=False)

    st_set = STBlockDataset(st_x_memmap, block2spots, st_coords)
    sm_set = SMDataset(sm_x, sm_label)

    metadata = {
        "n_spots": n_spots,
        "n_genes": n_genes,
        "block_size": float(block_size),
        "n_blocks": len(block2spots),
        "max_block_spots": max(len(v) for v in block2spots.values()),
        "celltype_names": celltype_names,
        **block_meta,
    }

    return st_ad, sc_ad,sm_ad, st_set, sm_set, celltype_names, metadata
