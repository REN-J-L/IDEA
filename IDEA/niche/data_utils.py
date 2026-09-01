from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gc
import json
import re

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from torch.utils.data import Dataset


# =============================================================================
# Basic helpers
# =============================================================================


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))


def _is_pathlike(x) -> bool:
    return isinstance(x, (str, Path))


def _is_path_input(st_ad) -> bool:
    """Return True when st_ad is one path or a list/tuple containing only paths."""
    if _is_pathlike(st_ad):
        return True
    if isinstance(st_ad, (list, tuple)):
        if len(st_ad) == 0:
            raise ValueError("At least one input file is required.")
        flags = [_is_pathlike(x) for x in st_ad]
        if any(flags) and not all(flags):
            raise TypeError(
                "Do not mix AnnData objects and file paths in st_ad. "
                "Use either only AnnData objects or only .h5ad paths."
            )
        return all(flags)
    return False


def _normalize_paths(st_ad, sample_names=None):
    if _is_pathlike(st_ad):
        paths = [Path(st_ad)]
    elif isinstance(st_ad, (list, tuple)) and all(_is_pathlike(x) for x in st_ad):
        paths = [Path(x) for x in st_ad]
    else:
        raise TypeError("st_ad is not a path or a list/tuple of paths.")

    if len(paths) == 0:
        raise ValueError("At least one input file is required.")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input h5ad file does not exist: {path}")

    if sample_names is None:
        names = [p.stem for p in paths]
    elif isinstance(sample_names, str):
        if len(paths) != 1:
            raise ValueError(
                "A string sample_names is only valid for a single input file."
            )
        names = [sample_names]
    else:
        names = [str(x) for x in sample_names]
        if len(names) != len(paths):
            raise ValueError("sample_names must match the number of input files.")

    return paths, names

def _get_samples_from_merged_backed(
    adata,
    condition_key,
    sample_names=None,
):
    """
    Get sample identities from a backed merged AnnData.

    Parameters
    ----------
    adata
        Backed AnnData.

    condition_key
        adata.obs column defining individual sections.

    sample_names
        Optional sample subset/order. If None, infer automatically.
    """

    if condition_key is None:
        raise ValueError(
            "condition_key is required for a merged h5ad file."
        )

    if condition_key not in adata.obs:
        raise KeyError(
            f"{condition_key!r} not found in merged AnnData.obs."
        )

    condition = (
        adata.obs[condition_key]
        .astype(str)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Infer sample order
    # --------------------------------------------------------

    if sample_names is None:

        # Preserve categorical order when available
        series = adata.obs[condition_key]

        if hasattr(series.dtype, "categories"):
            sample_names = [
                str(x)
                for x in series.cat.categories
            ]
        else:
            # Preserve first appearance order
            sample_names = list(
                dict.fromkeys(
                    condition.tolist()
                )
            )

    else:

        if isinstance(sample_names, str):
            sample_names = [
                str(sample_names)
            ]
        else:
            sample_names = [
                str(x)
                for x in sample_names
            ]

        available = set(
            condition.tolist()
        )

        missing = [
            name
            for name in sample_names
            if name not in available
        ]

        if missing:
            raise ValueError(
                "The following sample_names are not present "
                f"in adata.obs[{condition_key!r}]: {missing}"
            )

    if len(sample_names) == 0:
        raise ValueError(
            "No samples were found in the merged h5ad."
        )

    return condition, sample_names

def _prepare_niche_data_from_merged_path(
    st_ad,
    output_dir="./IDEAN_output",
    sample_names=None,
    condition_key="sample_key",
    spatial_key="spatial",

    target=4096,
    max_spots=6000,
    min_spots=1000,

    sliding=False,
    stride=None,

    min_genes=None,
    min_cells=None,

    use_hvg=True,
    n_top_hvg=1000,
    hvg_flavor="seurat_v3",
    hvg_layer=None,

    retain_adata=False,
    retain_obs_columns=None,

    memmap_chunk=20000,
):
    """
    Prepare a merged h5ad containing multiple spatial sections.

    The merged h5ad is opened in backed mode. Individual sections are
    materialized one at a time according to ``condition_key``.

    This avoids loading the complete merged expression matrix into RAM.
    """

    # ========================================================
    # 0. Input
    # ========================================================

    path = Path(st_ad)

    if not path.exists():
        raise FileNotFoundError(
            f"Input h5ad does not exist: {path}"
        )

    if condition_key is None:
        raise ValueError(
            "condition_key must be provided for "
            "a merged multi-section h5ad."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Open merged file in backed mode
    # ========================================================

    print(
        "[IDEA-N] Merged-h5ad streaming mode enabled."
    )

    print(
        f"[IDEA-N] Source: {path}"
    )

    print(
        f"[IDEA-N] Splitting sections by "
        f"obs[{condition_key!r}]"
    )

    merged = sc.read_h5ad(
        path,
        backed="r",
    )

    try:

        (
            condition_values,
            sample_names,
        ) = _get_samples_from_merged_backed(
            merged,
            condition_key=condition_key,
            sample_names=sample_names,
        )

        n_samples = len(
            sample_names
        )

        print(
            f"[IDEA-N] Found {n_samples} sections:"
        )

        for name in sample_names:
            n = int(
                np.sum(
                    condition_values
                    == str(name)
                )
            )

            print(
                f"    {name}: {n:,} spots"
            )

        # ====================================================
        # Pass 1
        #
        # Determine shared genes after filtering.
        #
        # Only one section is loaded at a time.
        # ====================================================

        gene_sets = []

        for sample_name in sample_names:

            print(
                f"[IDEA-N] Reading genes for "
                f"{sample_name}..."
            )

            adata = _load_one_sample_from_merged(
                backed_adata=merged,
                condition_values=condition_values,
                sample_name=sample_name,
            )

            # ----------------------------------------------
            # Optional filtering
            # ----------------------------------------------

            _apply_filters_inplace(
                adata,
                min_genes=min_genes,
                min_cells=min_cells,
            )

            # Preserve your current IDEA-N gene filter
            sc.pp.filter_genes(
                adata,
                min_cells=10,
            )

            genes = set(
                adata.var_names.astype(str)
            )

            gene_sets.append(
                genes
            )

            del adata
            gc.collect()

        # ====================================================
        # Shared gene space
        # ====================================================

        if n_samples > 1:

            common_genes = sorted(
                set.intersection(
                    *gene_sets
                )
            )

        else:

            common_genes = sorted(
                gene_sets[0]
            )

        del gene_sets
        gc.collect()

        if len(common_genes) == 0:
            raise ValueError(
                "No shared genes were found "
                "across merged sections."
            )

        print(
            f"[IDEA-N] Number of common genes: "
            f"{len(common_genes)}"
        )

        # ====================================================
        # Pass 2
        #
        # HVG selection section by section.
        # ====================================================

        if use_hvg:

            hvg_union = set()

            for sample_name in sample_names:

                print(
                    f"[IDEA-N] Selecting HVGs for "
                    f"{sample_name}..."
                )

                adata = _load_one_sample_from_merged(
                    backed_adata=merged,
                    condition_values=condition_values,
                    sample_name=sample_name,
                )

                if "long" in str(sample_name):

                    if "annotation" in adata.obs:
                        adata.obs["domain"] = (
                            adata.obs["annotation"]
                        )

                    if "count" in adata.layers:
                        adata.X = (
                            adata.layers["count"]
                        )

                _apply_filters_inplace(
                    adata,
                    min_genes=min_genes,
                    min_cells=min_cells,
                )

                # ------------------------------------------
                # Common-gene view only; no .copy()
                # ------------------------------------------

                missing = [
                    gene
                    for gene in common_genes
                    if gene not in adata.var_names
                ]

                if missing:
                    raise RuntimeError(
                        f"{len(missing)} common genes "
                        f"are missing in {sample_name}."
                    )

                ad_view = adata[
                    :,
                    common_genes,
                ]

                n_top = min(
                    int(n_top_hvg),
                    int(ad_view.n_vars),
                )

                if n_top <= 0:
                    raise ValueError(
                        f"No genes available for "
                        f"HVG selection in {sample_name}."
                    )

                hvg_result = (
                    sc.pp.highly_variable_genes(
                        ad_view,
                        n_top_genes=n_top,
                        flavor=hvg_flavor,
                        layer=hvg_layer,
                        subset=False,
                        inplace=False,
                    )
                )

                if (
                    "highly_variable"
                    not in hvg_result.columns
                ):
                    raise RuntimeError(
                        f"HVG selection failed for "
                        f"{sample_name}."
                    )

                batch_hvgs = (
                    ad_view.var_names[
                        np.asarray(
                            hvg_result[
                                "highly_variable"
                            ],
                            dtype=bool,
                        )
                    ]
                    .tolist()
                )

                hvg_union.update(
                    batch_hvgs
                )

                print(
                    f"[IDEA-N] {sample_name}: "
                    f"{len(batch_hvgs)} HVGs"
                )

                del hvg_result
                del ad_view
                del adata

                gc.collect()

            gene_names = sorted(
                hvg_union
            )

            if len(gene_names) == 0:
                raise ValueError(
                    "The union of section-specific "
                    "HVGs is empty."
                )

            gene_selection = (
                "common_genes_hvg_union"
                if n_samples > 1
                else "single_section_hvg"
            )

            print(
                f"[IDEA-N] Final HVG union: "
                f"{len(gene_names)} genes"
            )

        else:

            gene_names = list(
                common_genes
            )

            gene_selection = (
                "common_genes"
                if n_samples > 1
                else "all_filtered_genes"
            )

            print(
                f"[IDEA-N] HVG disabled: "
                f"using {len(gene_names)} genes"
            )

        del common_genes
        gc.collect()

        # ====================================================
        # Pass 3
        #
        # Blocks + expression memmap
        # ====================================================

        adatas = []

        coords_list = []
        memmap_infos = []

        train_records = []
        eval_records = []

        per_sample_meta = []

        offsets = []

        offset = 0

        for sample_idx, sample_name in enumerate(
            sample_names
        ):

            print(
                f"[IDEA-N] Preparing "
                f"{sample_name}..."
            )

            # ================================================
            # Only current section enters RAM
            # ================================================

            adata = _load_one_sample_from_merged(
                backed_adata=merged,
                condition_values=condition_values,
                sample_name=sample_name,
            )

            # ----------------------------------------------
            # Dataset-specific expression source
            # ----------------------------------------------

            if "long" in str(sample_name):

                if "annotation" in adata.obs:
                    adata.obs["domain"] = (
                        adata.obs["annotation"]
                    )

                if "count" in adata.layers:
                    adata.X = (
                        adata.layers["count"]
                    )

            _apply_filters_inplace(
                adata,
                min_genes=min_genes,
                min_cells=min_cells,
            )

            # ----------------------------------------------
            # Guarantee final gene set
            # ----------------------------------------------

            missing = [
                gene
                for gene in gene_names
                if gene not in adata.var_names
            ]

            if missing:
                raise RuntimeError(
                    f"Final gene set is not available "
                    f"in {sample_name}. "
                    f"Examples: {missing[:10]}"
                )

            # ----------------------------------------------
            # Make sample identity explicit
            # ----------------------------------------------

            adata.obs[
                "sample_key"
            ] = str(
                sample_name
            )

            # =================================================
            # Spatial coordinates
            # =================================================

            coords = _coords_from_adata(
                adata,
                spatial_key,
            )

            coords_list.append(
                coords
            )

            offsets.append(
                offset
            )

            # =================================================
            # Spatial block size
            # =================================================

            safe_name = _safe_filename(
                sample_name
            )

            block_size = (
                auto_tune_block_size(
                    coords,
                    target=target,
                    max_spots=max_spots,
                )
            )

            # =================================================
            # Evaluation blocks
            # =================================================

            (
                eval_block2spots,
                block_id,
                eval_meta,
            ) = build_blocks(
                coords,
                block_size,
                min_spots=min_spots,
                max_spots=max_spots,
            )

            np.save(
                output_dir
                / f"{safe_name}_block_id.npy",
                block_id,
            )

            # =================================================
            # Stage-1 blocks
            # =================================================

            if sliding:

                (
                    train_block2spots,
                    coverage,
                    train_meta,
                ) = build_blocks_sliding(
                    coords,
                    block_size,
                    stride=stride,
                    min_spots=min_spots,
                    max_spots=max_spots,
                )

                if np.any(
                    coverage == 0
                ):

                    print(
                        f"[IDEA-N] Warning: "
                        f"{(coverage == 0).sum()} "
                        f"spots in {sample_name} "
                        f"are not included in "
                        f"Stage-1 sliding blocks."
                    )

            else:

                train_block2spots = (
                    eval_block2spots
                )

                train_meta = dict(
                    eval_meta
                )

                train_meta[
                    "stride"
                ] = None

            # =================================================
            # Expression -> disk memmap
            #
            # No adata[:, gene_names].copy()
            # =================================================

            info = _write_memmap(
                adata,
                output_dir
                / f"{safe_name}_X_memmap.dat",
                gene_names=gene_names,
                chunk=memmap_chunk,
            )

            memmap_infos.append(
                info
            )

            np.save(
                output_dir
                / f"{safe_name}_coords.npy",
                coords,
            )

            # =================================================
            # Block records
            # =================================================

            _append_block_records(
                train_records,
                train_block2spots,
                sample_idx=sample_idx,
                offset=offset,
            )

            _append_block_records(
                eval_records,
                eval_block2spots,
                sample_idx=sample_idx,
                offset=offset,
            )

            n_obs = int(
                adata.n_obs
            )

            # =================================================
            # Metadata
            # =================================================

            per_sample_meta.append(
                {
                    "sample_name":
                        str(sample_name),

                    "source_h5ad":
                        str(path),

                    "source_condition_key":
                        str(condition_key),

                    "condition_id":
                        int(sample_idx),

                    "n_spots":
                        n_obs,

                    "n_genes":
                        int(
                            len(
                                gene_names
                            )
                        ),

                    "block_size":
                        float(
                            block_size
                        ),

                    "n_train_blocks":
                        int(
                            len(
                                train_block2spots
                            )
                        ),

                    "n_eval_blocks":
                        int(
                            len(
                                eval_block2spots
                            )
                        ),

                    "max_train_block_spots":
                        int(
                            max(
                                len(v)
                                for v
                                in train_block2spots.values()
                            )
                        ),

                    "max_eval_block_spots":
                        int(
                            max(
                                len(v)
                                for v
                                in eval_block2spots.values()
                            )
                        ),

                    "train_block_meta":
                        train_meta,

                    "eval_block_meta":
                        eval_meta,

                    "expression_memmap":
                        dict(info),
                }
            )

            # =================================================
            # What model.adatas retains
            # =================================================

            if retain_adata:

                # ------------------------------------------
                # Compatibility mode:
                # keeps expression and therefore consumes RAM
                # ------------------------------------------

                kept = (
                    adata[
                        :,
                        gene_names,
                    ]
                    .copy()
                )

                kept.obs[
                    "sample_key"
                ] = str(
                    sample_name
                )

                adatas.append(
                    kept
                )

                del kept

            else:

                # ------------------------------------------
                # Recommended large-data mode
                # ------------------------------------------

                light = (
                    _make_lightweight_adata(
                        adata=adata,
                        gene_names=gene_names,
                        spatial_key=spatial_key,
                        coords=coords,
                        sample_name=sample_name,
                        memmap_info=info,
                        retain_obs_columns=(
                            retain_obs_columns
                        ),
                    )
                )

                light.uns[
                    "IDEAN_source_h5ad"
                ] = str(
                    path
                )

                light.uns[
                    "IDEAN_source_condition_key"
                ] = str(
                    condition_key
                )

                adatas.append(
                    light
                )

                del light

            # =================================================
            # Global offset
            # =================================================

            offset += n_obs

            # =================================================
            # Release current section
            # =================================================

            del train_block2spots
            del eval_block2spots
            del block_id

            if sliding:
                del coverage

            del adata

            gc.collect()

        # ====================================================
        # Condition metadata
        # ====================================================

        condition_mapping = {
            str(name): int(i)
            for i, name
            in enumerate(
                sample_names
            )
        }

        metadata = {
            "input_mode":
                "merged_path_streaming",

            "source_h5ad":
                str(path),

            "retain_adata":
                bool(retain_adata),

            "n_samples":
                int(n_samples),

            "n_conditions":
                int(n_samples),

            "sample_names":
                [
                    str(x)
                    for x
                    in sample_names
                ],

            "condition_names":
                [
                    str(x)
                    for x
                    in sample_names
                ],

            "condition_mapping":
                condition_mapping,

            "condition_key":
                str(condition_key),

            "n_spots":
                int(offset),

            "n_genes":
                int(
                    len(
                        gene_names
                    )
                ),

            "gene_selection":
                gene_selection,

            "n_top_hvg":
                (
                    int(n_top_hvg)
                    if use_hvg
                    else None
                ),

            "hvg_flavor":
                (
                    str(hvg_flavor)
                    if use_hvg
                    else None
                ),

            "hvg_layer":
                hvg_layer,

            "n_train_blocks":
                int(
                    len(
                        train_records
                    )
                ),

            "n_eval_blocks":
                int(
                    len(
                        eval_records
                    )
                ),

            "target":
                int(target),

            "min_spots":
                int(min_spots),

            "max_spots":
                int(max_spots),

            "sliding":
                bool(sliding),

            "stride":
                (
                    None
                    if stride is None
                    else float(stride)
                ),

            "memmap_chunk":
                int(
                    memmap_chunk
                ),

            "samples":
                per_sample_meta,
        }

        # ====================================================
        # Save metadata
        # ====================================================

        with open(
            output_dir
            / "niche_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        # ====================================================
        # Dataset
        # ====================================================

        train_set = NicheBlockDataset(
            memmap_infos,
            coords_list,
            train_records,
        )

        eval_set = NicheBlockDataset(
            memmap_infos,
            coords_list,
            eval_records,
        )

        return (
            adatas,
            train_set,
            eval_set,
            gene_names,
            offsets,
            metadata,
        )

    finally:

        _close_backed(
            merged
        )

        del merged

        gc.collect()

def _load_one_sample_from_merged(
    backed_adata,
    condition_values,
    sample_name,
):
    """
    Materialize only one section from a backed merged AnnData.

    The complete merged expression matrix is never loaded into RAM.
    """

    idx = np.flatnonzero(
        condition_values == str(sample_name)
    )

    if len(idx) == 0:
        raise ValueError(
            f"No observations found for sample {sample_name!r}."
        )

    # --------------------------------------------------------
    # Critical:
    # backed_adata stays disk-backed;
    # only the current section is materialized.
    # --------------------------------------------------------

    adata = backed_adata[
        idx,
        :
    ].to_memory()

    return adata

def _coords_from_adata(adata, key):
    if key not in adata.obsm:
        raise KeyError(f"{key!r} not found in adata.obsm.")
    coords = np.asarray(adata.obsm[key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm[{key!r}] must have shape (N, >=2).")
    return coords[:, :2]


def _apply_filters_inplace(adata, min_genes=None, min_cells=None):
    """Filter an AnnData that was loaded inside IDEA-N.

    No explicit AnnData.copy() is made here. Scanpy may allocate temporary arrays
    internally, but the complete input object is not deliberately duplicated.
    """
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=int(min_genes))
    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=int(min_cells))
    return adata


def _close_backed(adata):
    try:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    except Exception:
        pass


# =============================================================================
# Spatial blocks
# =============================================================================


def split_oversized_blocks(block2spots, coords, max_spots):
    """Recursively bisect oversized blocks along their widest spatial axis."""
    new_block2spots = {}
    new_id = 0

    for spots in block2spots.values():
        spots = np.asarray(spots, dtype=np.int32)
        stack = [spots]
        while stack:
            idx = stack.pop()
            if len(idx) <= max_spots:
                new_block2spots[new_id] = idx
                new_id += 1
                continue

            sub_coords = coords[idx]
            span = sub_coords.max(axis=0) - sub_coords.min(axis=0)
            axis = int(np.argmax(span))
            order = np.argsort(sub_coords[:, axis])
            idx_sorted = idx[order]
            mid = len(idx_sorted) // 2
            stack.append(idx_sorted[:mid])
            stack.append(idx_sorted[mid:])

    return new_block2spots


def merge_small_blocks_with_capacity(block2spots, coords, min_spots, max_spots):
    """Merge small blocks into the nearest compatible block without exceeding max_spots."""
    block2spots = {
        b: np.asarray(v, dtype=np.int32) for b, v in block2spots.items()
    }

    changed = True
    while changed:
        changed = False
        block_ids = list(block2spots.keys())
        sizes = {b: len(block2spots[b]) for b in block_ids}
        small_blocks = sorted(
            [b for b in block_ids if sizes[b] < min_spots],
            key=lambda b: sizes[b],
        )
        if not small_blocks:
            break

        for sb in small_blocks:
            if sb not in block2spots:
                continue

            s_idx = block2spots[sb]
            s_size = len(s_idx)
            s_center = coords[s_idx].mean(axis=0)
            candidates = []

            for tb in list(block2spots.keys()):
                if tb == sb:
                    continue
                t_size = len(block2spots[tb])
                if s_size + t_size <= max_spots:
                    t_center = coords[block2spots[tb]].mean(axis=0)
                    candidates.append(
                        (float(np.linalg.norm(s_center - t_center)), tb)
                    )

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[0])
            target = candidates[0][1]
            block2spots[target] = np.concatenate(
                [block2spots[target], block2spots[sb]]
            )
            del block2spots[sb]
            changed = True

    return block2spots


def build_blocks(coords, block_size, min_spots=1000, max_spots=8000):
    """Build non-overlapping blocks with a soft lower and hard upper size bound."""
    if min_spots > max_spots:
        raise ValueError("min_spots must not exceed max_spots.")

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")
    if coords.shape[0] == 0:
        raise ValueError("coords contains no spots.")

    xy0 = coords.min(axis=0)
    grid = np.floor((coords - xy0) / block_size).astype(np.int32)
    grid_y = int(grid[:, 1].max()) + 1
    raw_id = grid[:, 0] * grid_y + grid[:, 1]

    block2spots = defaultdict(list)
    for i, b in enumerate(raw_id):
        block2spots[int(b)].append(i)

    block2spots = split_oversized_blocks(block2spots, coords, max_spots)
    block2spots = merge_small_blocks_with_capacity(
        block2spots,
        coords,
        min_spots=min_spots,
        max_spots=max_spots,
    )
    block2spots = split_oversized_blocks(block2spots, coords, max_spots)

    compact = {}
    block_id = np.empty(coords.shape[0], dtype=np.int32)
    for new_id, idx in enumerate(block2spots.values()):
        idx = np.asarray(idx, dtype=np.int32)
        compact[new_id] = idx
        block_id[idx] = new_id

    sizes = [len(v) for v in compact.values()]
    if not sizes:
        raise ValueError("No non-overlapping spatial blocks were generated.")

    meta = {
        "block_size": float(block_size),
        "min_spots": int(min_spots),
        "max_spots": int(max_spots),
        "n_blocks": int(len(compact)),
        "min_block_spots": int(min(sizes)),
        "max_block_spots": int(max(sizes)),
    }
    return compact, block_id, meta


def build_blocks_sliding(
    coords,
    block_size,
    stride=None,
    min_spots=1000,
    max_spots=6000,
):
    """Build overlapping sliding-window blocks and explicitly track coverage.

    The window placement follows the previous research implementation.  The
    current default uses 50% overlap (stride = block_size / 2).
    """
    coords = np.asarray(coords, dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")
    if coords.shape[0] == 0:
        raise ValueError("coords contains no spots.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if stride is None:
        stride = block_size / 1.2
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if min_spots <= 0:
        raise ValueError("min_spots must be positive.")
    if max_spots <= 0:
        raise ValueError("max_spots must be positive.")

    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_starts = np.arange(
        x_min,
        x_max - block_size + stride,
        stride,
        dtype=np.float32,
    )
    y_starts = np.arange(
        y_min,
        y_max - block_size + stride,
        stride,
        dtype=np.float32,
    )

    if len(x_starts) == 0:
        x_starts = np.asarray([x_min], dtype=np.float32)
    if len(y_starts) == 0:
        y_starts = np.asarray([y_min], dtype=np.float32)

    raw = {}
    bid = 0

    for xs in x_starts:
        xe = xs + block_size
        x_mask = (x >= xs) & (x < xe)
        if not np.any(x_mask):
            continue

        for ys in y_starts:
            ye = ys + block_size
            idx = np.where(x_mask & (y >= ys) & (y < ye))[0]
            if len(idx) >= min_spots:
                raw[bid] = idx.astype(np.int32)
                bid += 1

    if not raw:
        raise ValueError(
            "No valid sliding blocks were generated. "
            "Try decreasing min_spots or increasing block_size."
        )

    raw = split_oversized_blocks(raw, coords, max_spots)
    compact = {
        i: np.asarray(v, dtype=np.int32)
        for i, v in enumerate(raw.values())
    }

    coverage = np.zeros(coords.shape[0], dtype=np.int32)
    for idx in compact.values():
        coverage[idx] += 1

    meta = {
        "block_size": float(block_size),
        "stride": float(stride),
        "min_spots": int(min_spots),
        "max_spots": int(max_spots),
        "n_blocks": int(len(compact)),
        "n_uncovered": int((coverage == 0).sum()),
        "max_memberships": int(coverage.max()) if coverage.size else 0,
        "mean_memberships": float(coverage.mean()) if coverage.size else 0.0,
    }
    return compact, coverage, meta


def estimate_block_size(coords, target_spots_per_block, safety_factor=1.0):
    coords = np.asarray(coords, dtype=np.float64)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    area = (xmax - xmin) * (ymax - ymin)
    if area <= 0:
        raise ValueError("Spatial coordinates must span a non-zero 2D area.")
    density = coords.shape[0] / area
    return float(np.sqrt(target_spots_per_block / density) * safety_factor)


def evaluate_block_size(coords, block_size):
    coords = np.asarray(coords)
    xy0 = coords.min(axis=0)
    grid = np.floor((coords - xy0) / block_size).astype(np.int32)
    _, counts = np.unique(grid, axis=0, return_counts=True)
    return {
        "n_blocks": int(len(counts)),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "std": float(counts.std()),
        "min": int(counts.min()),
        "max": int(counts.max()),
    }


def auto_tune_block_size(
    coords,
    target,
    max_spots,
    tol=0.1,
    max_iter=10,
    safety=0.95,
):
    bs = estimate_block_size(coords, target)
    for _ in range(max_iter):
        stats = evaluate_block_size(coords, bs)
        if stats["max"] > max_spots:
            bs *= np.sqrt(max_spots / stats["max"]) * safety
            continue
        ratio = stats["mean"] / target
        if abs(ratio - 1.0) < tol:
            break
        bs /= np.sqrt(ratio)
    return float(bs)


# =============================================================================
# Expression memmap
# =============================================================================


def _write_memmap(
    adata,
    path,
    gene_names=None,
    chunk=20000,
):
    """Write selected expression columns directly to a dense disk memmap.

    Important for large data:
    -------------------------
    This function DOES NOT create ``adata[:, gene_names].copy()``.  Instead it
    finds the selected column indices and writes row chunks directly from the
    original expression matrix.  Therefore only one dense chunk is materialized
    at a time when the source matrix is sparse.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = int(adata.n_obs)
    if gene_names is None:
        gene_names = adata.var_names.tolist()
        gene_idx = np.arange(adata.n_vars, dtype=np.int64)
    else:
        gene_names = [str(x) for x in gene_names]
        gene_idx = adata.var_names.get_indexer(gene_names)
        if np.any(gene_idx < 0):
            bad = np.where(gene_idx < 0)[0]
            missing = [gene_names[i] for i in bad[:10]]
            raise KeyError(
                f"Selected genes are missing from AnnData. Examples: {missing}"
            )
        gene_idx = gene_idx.astype(np.int64, copy=False)

    g = len(gene_names)
    if n == 0:
        raise ValueError(f"Cannot create memmap for empty AnnData: {path}")
    if g == 0:
        raise ValueError(f"Cannot create memmap with zero selected genes: {path}")

    mm = np.memmap(
        path,
        dtype=np.float32,
        mode="w+",
        shape=(n, g),
    )

    X = adata.X
    for start in range(0, n, int(chunk)):
        end = min(start + int(chunk), n)
        block = X[start:end, :]
        block = block[:, gene_idx]
        if sparse.issparse(block):
            block = block.toarray()
        mm[start:end] = np.asarray(block, dtype=np.float32)
        del block

    mm.flush()
    del mm

    return {
        "path": str(path),
        "n": n,
        "g": int(g),
        "dtype": "float32",
    }


# =============================================================================
# Block dataset
# =============================================================================


class NicheBlockDataset(Dataset):
    """Return one spatial block at a time from disk-backed expression memmaps."""

    def __init__(self, memmap_infos, coords_list, block_records):
        self.memmap_infos = list(memmap_infos)
        self.coords_list = [np.asarray(c, dtype=np.float32) for c in coords_list]
        self.block_records = list(block_records)
        self.X_mms = [
            np.memmap(
                info["path"],
                dtype=np.float32,
                mode="r",
                shape=(info["n"], info["g"]),
            )
            for info in self.memmap_infos
        ]

    def __len__(self):
        return len(self.block_records)

    def __getitem__(self, idx):
        rec = self.block_records[idx]
        sample_idx = int(rec["sample_idx"])
        local_idx = np.asarray(rec["local_idx"], dtype=np.int64)
        global_idx = np.asarray(rec["global_idx"], dtype=np.int64)

        # Fancy indexing on np.memmap produces the current block only.
        x = torch.from_numpy(
            np.asarray(
                self.X_mms[sample_idx][local_idx],
                dtype=np.float32,
            ).copy()
        )
        pos = torch.from_numpy(
            self.coords_list[sample_idx][local_idx].copy()
        )
        condition_id = torch.full(
            (len(local_idx),),
            sample_idx,
            dtype=torch.long,
        )

        return (
            x,
            pos,
            torch.as_tensor(global_idx, dtype=torch.long),
            condition_id,
        )


# =============================================================================
# Path-streaming gene selection
# =============================================================================


def _collect_filtered_gene_sets_from_paths(
    paths,
    sample_names,
    min_genes=None,
    min_cells=None,
):
    """Collect per-file gene sets while keeping at most one full AnnData in RAM."""
    gene_sets = []

    use_backed_metadata_only = min_genes is None and min_cells is None

    for path, sample_name in zip(paths, sample_names):
        print(f"[IDEA-N] Reading gene metadata for {sample_name}...")

        if use_backed_metadata_only:
            adata = sc.read_h5ad(path)
            sc.pp.filter_genes(adata, min_cells=10)

            genes = set(adata.var_names.astype(str))
            _close_backed(adata)
            del adata
        else:
            adata = sc.read_h5ad(path)
            _apply_filters_inplace(
                adata,
                min_genes=min_genes,
                min_cells=min_cells,
            )
            sc.pp.filter_genes(adata, min_cells=10)


            genes = set(adata.var_names.astype(str))
            del adata
            gc.collect()

        gene_sets.append(genes)

    return gene_sets


def _select_hvg_union_from_paths(
    paths,
    sample_names,
    common_genes,
    n_top_hvg=1000,
    hvg_flavor="seurat_v3",
    hvg_layer=None,
    min_genes=None,
    min_cells=None,
):
    """Select HVGs one section at a time without retaining complete sections."""
    hvg_union = set()
    common_genes = list(common_genes)

    for path, sample_name in zip(paths, sample_names):
        print(f"[IDEA-N] Selecting HVGs for {sample_name}...")

        adata = sc.read_h5ad(path)



        _apply_filters_inplace(
            adata,
            min_genes=min_genes,
            min_cells=min_cells,
        )

        missing = [g for g in common_genes if g not in adata.var_names]
        if missing:
            raise RuntimeError(
                f"Common-gene consistency failed for {sample_name}; "
                f"{len(missing)} common genes are missing after filtering."
            )

        # A view is sufficient because highly_variable_genes(..., inplace=False)
        # reads the matrix but does not need to write HVG flags into the AnnData.
        ad_view = adata[:, common_genes]

        n_top = min(int(n_top_hvg), int(ad_view.n_vars))
        if n_top <= 0:
            raise ValueError(f"No genes available for HVG selection in {sample_name}.")

        hvg_result = sc.pp.highly_variable_genes(
            ad_view,
            n_top_genes=n_top,
            flavor=hvg_flavor,
            subset=False,
            inplace=False,

        )

        if "highly_variable" not in hvg_result.columns:
            raise RuntimeError(f"HVG selection failed for {sample_name}.")

        batch_hvgs = ad_view.var_names[
            np.asarray(hvg_result["highly_variable"], dtype=bool)
        ].tolist()
        hvg_union.update(batch_hvgs)

        print(f"[IDEA-N] {sample_name}: {len(batch_hvgs)} HVGs")

        del hvg_result
        del ad_view
        del adata
        gc.collect()

    gene_names = sorted(hvg_union)
    if not gene_names:
        raise ValueError("The union of section-specific HVGs is empty.")
    return gene_names


# =============================================================================
# Lightweight AnnData retained by the high-level model
# =============================================================================


def _make_lightweight_adata(
    adata,
    gene_names,
    spatial_key,
    coords,
    sample_name,
    memmap_info,
    retain_obs_columns=None,
):
    """Create an AnnData that keeps metadata but not the complete expression matrix.

    ``X`` is a sparse all-zero placeholder.  The real expression used for
    training is stored in ``memmap_info['path']`` and consumed by
    ``NicheBlockDataset``.  This prevents ``model.adatas`` from keeping a second
    complete expression matrix in RAM.
    """
    if retain_obs_columns is None:
        obs = adata.obs.copy()
    else:
        missing = [c for c in retain_obs_columns if c not in adata.obs.columns]
        if missing:
            raise KeyError(f"retain_obs_columns not found in adata.obs: {missing}")
        obs = adata.obs.loc[:, list(retain_obs_columns)].copy()

    obs["sample_key"] = str(sample_name)

    # Retain selected-gene metadata only.  This is small relative to X.
    var = adata.var.loc[list(gene_names)].copy()

    light = sc.AnnData(
        X=sparse.csr_matrix(
            (adata.n_obs, len(gene_names)),
            dtype=np.float32,
        ),
        obs=obs,
        var=var,
    )
    light.obsm[spatial_key] = np.asarray(coords, dtype=np.float32).copy()
    light.uns["IDEAN_lightweight"] = True
    light.uns["IDEAN_expression_memmap"] = dict(memmap_info)
    light.uns["IDEAN_source_h5ad"] = str(getattr(adata, "filename", "") or "")
    return light


# =============================================================================
# Shared record builder
# =============================================================================


def _append_block_records(container, block2spots, sample_idx, offset):
    for local_idx in block2spots.values():
        local_idx = np.asarray(local_idx, dtype=np.int64)
        container.append(
            {
                "sample_idx": int(sample_idx),
                "local_idx": local_idx,
                "global_idx": local_idx + int(offset),
            }
        )


# =============================================================================
# Memory-efficient path mode
# =============================================================================


def _prepare_niche_data_from_paths(
    st_ad,
    output_dir="./IDEAN_output",
    sample_names=None,
    condition_key=None,
    spatial_key="spatial",
    target=4096,
    max_spots=6000,
    min_spots=1000,
    sliding=False,
    stride=None,
    min_genes=None,
    min_cells=None,
    use_hvg=True,
    n_top_hvg=1000,
    hvg_flavor="seurat_v3",
    hvg_layer=None,
    retain_adata=False,
    retain_obs_columns=None,
    memmap_chunk=20000,
):
    """Prepare IDEA-N from .h5ad paths with one full section in RAM at a time."""
    if condition_key is not None:
        raise ValueError(
            "Memory-efficient path mode assumes one condition/section per h5ad file. "
            "For a concatenated h5ad with condition_key, either split it into files "
            "first or pass an AnnData object instead."
        )

    paths, sample_names = _normalize_paths(st_ad, sample_names=sample_names)
    n_samples = len(paths)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[IDEA-N] Path-streaming mode enabled: {n_samples} section(s). "
        "Only one complete h5ad is kept in RAM at a time."
    )

    # -------------------------------------------------------------------------
    # Pass 1: shared genes
    # -------------------------------------------------------------------------
    gene_sets = _collect_filtered_gene_sets_from_paths(
        paths,
        sample_names,
        min_genes=min_genes,
        min_cells=min_cells,
    )

    if n_samples > 1:
        common_genes = sorted(set.intersection(*gene_sets))
    else:
        common_genes = sorted(gene_sets[0])

    del gene_sets
    gc.collect()

    if not common_genes:
        raise ValueError("No shared genes were found across sections.")

    print(f"[IDEA-N] Number of common genes: {len(common_genes)}")

    # -------------------------------------------------------------------------
    # Pass 2: HVG union (one section at a time)
    # -------------------------------------------------------------------------
    if use_hvg:
        gene_names = _select_hvg_union_from_paths(
            paths=paths,
            sample_names=sample_names,
            common_genes=common_genes,
            n_top_hvg=n_top_hvg,
            hvg_flavor=hvg_flavor,
            hvg_layer=hvg_layer,
            min_genes=min_genes,
            min_cells=min_cells,
        )
        gene_selection = (
            "common_genes_hvg_union" if n_samples > 1 else "single_section_hvg"
        )
        print(f"[IDEA-N] Final HVG union: {len(gene_names)} genes")
    else:
        gene_names = common_genes
        gene_selection = (
            "common_genes" if n_samples > 1 else "all_filtered_genes"
        )
        print(f"[IDEA-N] HVG disabled: using {len(gene_names)} genes")

    # common_genes is no longer needed after final gene_names are known.
    del common_genes
    gc.collect()

    # -------------------------------------------------------------------------
    # Pass 3: blocks + direct expression memmap, one section at a time
    # -------------------------------------------------------------------------
    adatas = []
    coords_list = []
    memmap_infos = []
    train_records = []
    eval_records = []
    per_sample_meta = []
    offsets = []
    offset = 0

    for sample_idx, (path, sample_name) in enumerate(zip(paths, sample_names)):
        print(f"[IDEA-N] Preparing {sample_name}...")

        adata = sc.read_h5ad(path)

        
        _apply_filters_inplace(
            adata,
            min_genes=min_genes,
            min_cells=min_cells,
        )

        # The selected-gene columns are written directly into a memmap later;
        # no adata[:, gene_names].copy() is created here.
        missing = [g for g in gene_names if g not in adata.var_names]
        if missing:
            raise RuntimeError(
                f"Final gene set is not available in {sample_name}. "
                f"Examples: {missing[:10]}"
            )

        coords = _coords_from_adata(adata, spatial_key)
        coords_list.append(coords)
        offsets.append(offset)

        safe_name = _safe_filename(sample_name)
        block_size = auto_tune_block_size(
            coords,
            target=target,
            max_spots=max_spots,
        )

        eval_block2spots, block_id, eval_meta = build_blocks(
            coords,
            block_size,
            min_spots=min_spots,
            max_spots=max_spots,
        )
        np.save(output_dir / f"{safe_name}_block_id.npy", block_id)

        if sliding:
            train_block2spots, coverage, train_meta = build_blocks_sliding(
                coords,
                block_size,
                stride=stride,
                min_spots=min_spots,
                max_spots=max_spots,
            )
            if np.any(coverage == 0):
                print(
                    f"[IDEA-N] Warning: {(coverage == 0).sum()} spots in "
                    f"{sample_name} are not present in Stage-1 sliding blocks. "
                    "They remain covered by eval blocks for latent inference, "
                    "Leiden, Stage 2, and prediction."
                )
        else:
            train_block2spots = eval_block2spots
            train_meta = dict(eval_meta)
            train_meta["stride"] = None

        info = _write_memmap(
            adata,
            output_dir / f"{safe_name}_X_memmap.dat",
            gene_names=gene_names,
            chunk=memmap_chunk,
        )
        memmap_infos.append(info)
        np.save(output_dir / f"{safe_name}_coords.npy", coords)

        _append_block_records(
            train_records,
            train_block2spots,
            sample_idx=sample_idx,
            offset=offset,
        )
        _append_block_records(
            eval_records,
            eval_block2spots,
            sample_idx=sample_idx,
            offset=offset,
        )

        n_obs = int(adata.n_obs)
        sample_meta = {
            "sample_name": str(sample_name),
            "source_h5ad": str(path),
            "condition_id": int(sample_idx),
            "n_spots": n_obs,
            "n_genes": int(len(gene_names)),
            "block_size": float(block_size),
            "n_train_blocks": int(len(train_block2spots)),
            "n_eval_blocks": int(len(eval_block2spots)),
            "max_train_block_spots": int(
                max(len(v) for v in train_block2spots.values())
            ),
            "max_eval_block_spots": int(
                max(len(v) for v in eval_block2spots.values())
            ),
            "train_block_meta": train_meta,
            "eval_block_meta": eval_meta,
            "expression_memmap": dict(info),
        }
        per_sample_meta.append(sample_meta)

        if retain_adata:
            # Compatibility mode only.  This creates a final selected-gene copy
            # and therefore forfeits most of the memory benefit of streaming.
            kept = adata[:, gene_names].copy()
            kept.obs["sample_key"] = str(sample_name)
            adatas.append(kept)
            del kept
        else:
            # Recommended large-data mode: keep metadata/coordinates only.
            light = _make_lightweight_adata(
                adata=adata,
                gene_names=gene_names,
                spatial_key=spatial_key,
                coords=coords,
                sample_name=sample_name,
                memmap_info=info,
                retain_obs_columns=retain_obs_columns,
            )
            light.uns["IDEAN_source_h5ad"] = str(path)
            adatas.append(light)
            del light

        offset += n_obs

        # Critical for bounded peak RAM: release the complete section before
        # reading the next h5ad file.
        del train_block2spots
        del eval_block2spots
        del block_id
        if sliding:
            del coverage
        del adata
        gc.collect()

    condition_mapping = {
        str(name): int(i) for i, name in enumerate(sample_names)
    }

    metadata = {
        "input_mode": "path_streaming",
        "retain_adata": bool(retain_adata),
        "n_samples": int(n_samples),
        "n_conditions": int(n_samples),
        "sample_names": [str(x) for x in sample_names],
        "condition_names": [str(x) for x in sample_names],
        "condition_mapping": condition_mapping,
        "condition_key": None,
        "n_spots": int(offset),
        "n_genes": int(len(gene_names)),
        "gene_selection": gene_selection,
        "n_top_hvg": int(n_top_hvg) if use_hvg else None,
        "hvg_flavor": str(hvg_flavor) if use_hvg else None,
        "hvg_layer": hvg_layer,
        "n_train_blocks": int(len(train_records)),
        "n_eval_blocks": int(len(eval_records)),
        "target": int(target),
        "min_spots": int(min_spots),
        "max_spots": int(max_spots),
        "sliding": bool(sliding),
        "stride": None if stride is None else float(stride),
        "memmap_chunk": int(memmap_chunk),
        "samples": per_sample_meta,
    }

    with open(
        output_dir / "niche_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    train_set = NicheBlockDataset(
        memmap_infos,
        coords_list,
        train_records,
    )
    eval_set = NicheBlockDataset(
        memmap_infos,
        coords_list,
        eval_records,
    )

    return (
        adatas,
        train_set,
        eval_set,
        gene_names,
        offsets,
        metadata,
    )


# =============================================================================
# AnnData input mode (backward-compatible)
# =============================================================================


def _normalize_input_adatas(
    st_ad,
    sample_names=None,
    condition_key=None,
    copy_input=True,
):
    """Normalize AnnData input.

    Path input is handled separately by ``_prepare_niche_data_from_paths``.
    """
    if isinstance(st_ad, (list, tuple)):
        if not st_ad:
            raise ValueError("At least one AnnData object is required.")
        adatas = [ad.copy() if copy_input else ad for ad in st_ad]
        if sample_names is None:
            sample_names = [f"sample_{i}" for i in range(len(adatas))]
        if len(sample_names) != len(adatas):
            raise ValueError("sample_names must match the number of AnnData objects.")
        return adatas, [str(x) for x in sample_names]

    adata = st_ad.copy() if copy_input else st_ad

    if condition_key is not None:
        if condition_key not in adata.obs:
            raise KeyError(f"{condition_key!r} not found in adata.obs.")
        condition = adata.obs[condition_key].astype("category")
        inferred_names = [str(x) for x in condition.cat.categories]

        # Splitting a single concatenated AnnData necessarily materializes each
        # condition so the parent matrix can be released independently.
        adatas = [
            adata[condition.astype(str) == name].copy()
            for name in inferred_names
        ]
        if sample_names is not None and [str(x) for x in sample_names] != inferred_names:
            raise ValueError(
                "When st_ad is a single AnnData and condition_key is used, "
                "sample_names must match the categorical condition order or be omitted."
            )
        return adatas, inferred_names

    if sample_names is None:
        single_name = "sample_0"
    elif isinstance(sample_names, str):
        single_name = sample_names
    else:
        if len(sample_names) != 1:
            raise ValueError(
                "A single AnnData without condition_key expects exactly one sample name."
            )
        single_name = str(sample_names[0])

    return [adata], [single_name]


def _prepare_niche_data_from_adatas(
    st_ad,
    output_dir="./IDEAN_output",
    sample_names=None,
    condition_key=None,
    spatial_key="spatial",
    target=4096,
    max_spots=6000,
    min_spots=1000,
    sliding=False,
    stride=None,
    min_genes=None,
    min_cells=None,
    use_hvg=True,
    n_top_hvg=1000,
    hvg_flavor="seurat_v3",
    hvg_layer=None,
    copy_input=True,
    memmap_chunk=20000,
):
    """Backward-compatible in-memory AnnData preparation."""
    adatas, sample_names = _normalize_input_adatas(
        st_ad,
        sample_names=sample_names,
        condition_key=condition_key,
        copy_input=copy_input,
    )
    n_samples = len(adatas)

    for i, adata in enumerate(adatas):
        _apply_filters_inplace(
            adata,
            min_genes=min_genes,
            min_cells=min_cells,
        )
        adata.obs["sample_key"] = str(sample_names[i])

    if n_samples > 1:
        gene_sets = [set(ad.var_names.astype(str)) for ad in adatas]
        common_genes = sorted(set.intersection(*gene_sets))
        if not common_genes:
            raise ValueError("No shared genes were found across sections.")
    else:
        common_genes = adatas[0].var_names.astype(str).tolist()

    if use_hvg:
        hvg_union = set()
        for adata, sample_name in zip(adatas, sample_names):
            ad_view = adata[:, common_genes]
            n_top = min(int(n_top_hvg), int(ad_view.n_vars))
            hvg_result = sc.pp.highly_variable_genes(
                ad_view,
                n_top_genes=n_top,
                flavor=hvg_flavor,
                layer=hvg_layer,
                subset=False,
                inplace=False,
            )
            batch_hvgs = ad_view.var_names[
                np.asarray(hvg_result["highly_variable"], dtype=bool)
            ].tolist()
            hvg_union.update(batch_hvgs)
            print(f"[IDEA-N] {sample_name}: {len(batch_hvgs)} HVGs")
        gene_names = sorted(hvg_union)
        gene_selection = (
            "common_genes_hvg_union" if n_samples > 1 else "single_section_hvg"
        )
    else:
        gene_names = sorted(common_genes)
        gene_selection = (
            "common_genes" if n_samples > 1 else "all_filtered_genes"
        )

    # One final selected-gene object per sample is kept because the user passed
    # AnnData objects and the historical API exposes model.adatas.
    adatas = [ad[:, gene_names].copy() for ad in adatas]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_list = []
    memmap_infos = []
    train_records = []
    eval_records = []
    per_sample_meta = []
    offsets = []
    offset = 0

    for sample_idx, (adata, sample_name) in enumerate(zip(adatas, sample_names)):
        coords = _coords_from_adata(adata, spatial_key)
        coords_list.append(coords)
        offsets.append(offset)

        safe_name = _safe_filename(sample_name)
        block_size = auto_tune_block_size(
            coords,
            target=target,
            max_spots=max_spots,
        )

        eval_block2spots, block_id, eval_meta = build_blocks(
            coords,
            block_size,
            min_spots=min_spots,
            max_spots=max_spots,
        )
        np.save(output_dir / f"{safe_name}_block_id.npy", block_id)

        if sliding:
            train_block2spots, coverage, train_meta = build_blocks_sliding(
                coords,
                block_size,
                stride=stride,
                min_spots=min_spots,
                max_spots=max_spots,
            )
        else:
            train_block2spots = eval_block2spots
            train_meta = dict(eval_meta)
            train_meta["stride"] = None

        info = _write_memmap(
            adata,
            output_dir / f"{safe_name}_X_memmap.dat",
            gene_names=None,
            chunk=memmap_chunk,
        )
        memmap_infos.append(info)
        np.save(output_dir / f"{safe_name}_coords.npy", coords)

        _append_block_records(
            train_records,
            train_block2spots,
            sample_idx,
            offset,
        )
        _append_block_records(
            eval_records,
            eval_block2spots,
            sample_idx,
            offset,
        )

        per_sample_meta.append(
            {
                "sample_name": str(sample_name),
                "condition_id": int(sample_idx),
                "n_spots": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "block_size": float(block_size),
                "n_train_blocks": int(len(train_block2spots)),
                "n_eval_blocks": int(len(eval_block2spots)),
                "max_train_block_spots": int(
                    max(len(v) for v in train_block2spots.values())
                ),
                "max_eval_block_spots": int(
                    max(len(v) for v in eval_block2spots.values())
                ),
                "train_block_meta": train_meta,
                "eval_block_meta": eval_meta,
                "expression_memmap": dict(info),
            }
        )

        offset += int(adata.n_obs)

    condition_mapping = {
        str(name): int(i) for i, name in enumerate(sample_names)
    }
    metadata = {
        "input_mode": "anndata",
        "n_samples": int(n_samples),
        "n_conditions": int(n_samples),
        "sample_names": [str(x) for x in sample_names],
        "condition_names": [str(x) for x in sample_names],
        "condition_mapping": condition_mapping,
        "condition_key": condition_key,
        "n_spots": int(offset),
        "n_genes": int(len(gene_names)),
        "gene_selection": gene_selection,
        "n_top_hvg": int(n_top_hvg) if use_hvg else None,
        "hvg_flavor": str(hvg_flavor) if use_hvg else None,
        "hvg_layer": hvg_layer,
        "n_train_blocks": int(len(train_records)),
        "n_eval_blocks": int(len(eval_records)),
        "target": int(target),
        "min_spots": int(min_spots),
        "max_spots": int(max_spots),
        "sliding": bool(sliding),
        "stride": None if stride is None else float(stride),
        "memmap_chunk": int(memmap_chunk),
        "samples": per_sample_meta,
    }

    with open(
        output_dir / "niche_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    train_set = NicheBlockDataset(
        memmap_infos,
        coords_list,
        train_records,
    )
    eval_set = NicheBlockDataset(
        memmap_infos,
        coords_list,
        eval_records,
    )

    return (
        adatas,
        train_set,
        eval_set,
        gene_names,
        offsets,
        metadata,
    )


# =============================================================================
# Public entry point
# =============================================================================


def prepare_niche_data(
    st_ad,
    output_dir="./IDEAN_output",
    sample_names=None,
    condition_key=None,
    spatial_key="spatial",
    target=4096,
    max_spots=6000,
    min_spots=1000,
    sliding=False,
    stride=None,
    min_genes=None,
    min_cells=None,
    use_hvg=True,
    n_top_hvg=1000,
    hvg_flavor="seurat_v3",
    hvg_layer=None,
    # ---------------------------------------------------------------------
    # Memory controls
    # ---------------------------------------------------------------------
    retain_adata=True,
    retain_obs_columns=None,
    copy_input=True,
    memmap_chunk=20000,
):
    """Prepare single- or multi-section data for IDEA-N.

    Parameters
    ----------
    st_ad
        Either:
        1. one AnnData;
        2. a list/tuple of AnnData objects;
        3. one .h5ad file path; or
        4. a list/tuple of .h5ad file paths.

        Path input automatically uses the memory-efficient streaming workflow:
        each complete section is read, processed, written to the expression
        memmap, and released before the next section is loaded.

    retain_adata
        Relevant to path input.  False (recommended for large data) keeps only
        a lightweight metadata AnnData in ``model.adatas``.  True keeps a full
        selected-gene AnnData for every section and therefore uses substantially
        more RAM.

    retain_obs_columns
        Optional list of obs columns retained in lightweight AnnData objects.
        None retains all obs columns.  Expression X is never retained in
        lightweight mode.

    copy_input
        Relevant to AnnData input.  True protects user-provided AnnData objects
        from in-place filtering.  Path input ignores this because the function
        owns the objects it reads from disk.

    memmap_chunk
        Number of spots written per expression-memmap chunk.

    Notes
    -----
    For multi-file path input, one file should correspond to one section or
    condition.  This is the most memory-efficient mode and is recommended for
    large multi-section integration.
    """
    # ============================================================
# Mode 1:
# one merged h5ad path + condition_key
# ============================================================

    if (
        _is_pathlike(st_ad)
        and condition_key is not None
    ):

        return _prepare_niche_data_from_merged_path(
            st_ad=st_ad,

            output_dir=output_dir,

            sample_names=sample_names,
            condition_key=condition_key,
            spatial_key=spatial_key,

            target=target,
            max_spots=max_spots,
            min_spots=min_spots,

            sliding=sliding,
            stride=stride,

            min_genes=min_genes,
            min_cells=min_cells,

            use_hvg=use_hvg,
            n_top_hvg=n_top_hvg,
            hvg_flavor=hvg_flavor,
            hvg_layer=hvg_layer,

            retain_adata=retain_adata,
            retain_obs_columns=retain_obs_columns,

            memmap_chunk=memmap_chunk,
        )


    # ============================================================
    # Mode 2:
    # one h5ad per section, or list of section paths
    # ============================================================

    if _is_path_input(st_ad):

        return _prepare_niche_data_from_paths(
            st_ad=st_ad,

            output_dir=output_dir,

            sample_names=sample_names,

            # important:
            # per-file path mode does not split internally
            condition_key=None,

            spatial_key=spatial_key,

            target=target,
            max_spots=max_spots,
            min_spots=min_spots,

            sliding=sliding,
            stride=stride,

            min_genes=min_genes,
            min_cells=min_cells,

            use_hvg=use_hvg,
            n_top_hvg=n_top_hvg,
            hvg_flavor=hvg_flavor,
            hvg_layer=hvg_layer,

            retain_adata=retain_adata,
            retain_obs_columns=retain_obs_columns,

            memmap_chunk=memmap_chunk,
        )


    # ============================================================
    # Mode 3:
    # AnnData / list of AnnData
    # ============================================================

    return _prepare_niche_data_from_adatas(
        st_ad=st_ad,

        output_dir=output_dir,

        sample_names=sample_names,
        condition_key=condition_key,
        spatial_key=spatial_key,

        target=target,
        max_spots=max_spots,
        min_spots=min_spots,

        sliding=sliding,
        stride=stride,

        min_genes=min_genes,
        min_cells=min_cells,

        use_hvg=use_hvg,
        n_top_hvg=n_top_hvg,
        hvg_flavor=hvg_flavor,
        hvg_layer=hvg_layer,

        copy_input=copy_input,

        memmap_chunk=memmap_chunk,
    )