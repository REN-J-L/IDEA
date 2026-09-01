from __future__ import annotations

from . import data_utils
from .base_model import IDEACModel


def init_model(
    sc_ad,
    st_ad,
    celltype_key="scelltype",
    spatial_key = 'spatial',
    output_dir="./IDEAC_output",
    target=4096,
    max_spots=6000,
    min_spots=1000,
    n_pseudo=100000,
    n_top_markers=200,
    deg_method="t-test",
    log2fc_min=0.5,
    pval_cutoff=0.01,
    pct_min=0.1,
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
    only_use_common=True,
    cells_mean=1,
    cells_min=2,
    cells_max=3,
):
    """Initialize IDEA-C in a SPACEL/Spoint-like API.

    Parameters
    ----------
    sc_ad
        Single-cell reference AnnData.
    st_ad
        Spatial transcriptomics AnnData with ``obsm['spatial']``.
    celltype_key
        Cell-type annotation column in ``sc_ad.obs``.

    Returns
    -------
    IDEACModel
        High-level object exposing ``train_model()``, ``predict()``, ``save()``
        and ``load()``.
    """
    st_ad, sc_ad, sm_ad, st_set, sm_set, celltype_names, metadata = data_utils.prepare_data(
        sc_ad=sc_ad,
        st_ad=st_ad,
        celltype_key=celltype_key,
        spatial_key = spatial_key,
        output_dir=output_dir,
        target=target,
        max_spots=max_spots,
        min_spots=min_spots,
        n_pseudo=n_pseudo,
        n_top_markers=n_top_markers,
        deg_method=deg_method,
        log2fc_min=log2fc_min,
        pval_cutoff=pval_cutoff,
        pct_min=pct_min,
        only_use_common = only_use_common,
        cells_mean=cells_mean,
        cells_min=cells_min,
        cells_max=cells_max,
    )

    return IDEACModel(
        st_ad=st_ad,
        sc_ad=sc_ad,
        sm_ad=sm_ad,
        st_set=st_set,
        sm_set=sm_set,
        celltype_names=celltype_names,
        hidden_size=hidden_size,
        latent_size=latent_size,
        dropout=dropout,
        var_eps=var_eps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lkl=lkl,
        lmmd=lmmd,
        lmse=lmse,
        seed=seed,
        use_gpu=use_gpu,
        metadata=metadata,
    )
