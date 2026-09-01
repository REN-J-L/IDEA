from __future__ import annotations

from .base_model import IDEANModel
from .data_utils import prepare_niche_data


def init_model(
    st_ad,
    output_dir="./IDEAN_output",
    sample_names=None,
    condition_key=None,
    spatial_key='spatial',
    use_condition=None,

    use_domain_adversarial=False,
    domain_adv_weight=1.0,
    domain_adv_gamma=10.0,
    domain_label_smoothing=0.1,
    lr_domain=None,

    use_hvg = True,
    target=4096,
    max_spots=6000,
    min_spots=1000,
    n_top_hvg=2000,
    sliding=False,
    stride=None,
    min_genes=None,
    min_cells=None,
    hidden_size=512,
    latent_size=128,
    dropout=0.0,
    var_eps=1e-4,
    max_cluster=100,
    knn_k=25,
    spatial_weight=15.0,
    radius_scale=2.5,
    lr_encoder=5e-3,
    lr_decoder=5e-3,
    lr_classifier=1e-4,
    classifier_weight=10.0,
    seed=42,
    use_gpu=None,
):
    """Initialize IDEA-N.

    Parameters
    ----------
    st_ad
        One AnnData, a concatenated AnnData with ``condition_key``, or a list
        of AnnData objects (one per section/platform).
    condition_key
        Optional ``adata.obs`` column used to split a single concatenated
        AnnData into conditions.
    use_condition
        If None, conditional reconstruction is enabled automatically for
        multi-condition input and disabled for a single condition. Set False
        on multi-section data for a conditional-modeling ablation.

    Notes
    -----
    Condition is supplied only to the reconstruction decoder. KANA, encoder,
    joint Leiden clustering, and the niche classifier do not receive condition
    directly.
    """
    adatas, train_set, eval_set, gene_names, offsets, metadata = prepare_niche_data(
        st_ad=st_ad,
        output_dir=output_dir,
        sample_names=sample_names,
        condition_key=condition_key,
        spatial_key=spatial_key,
        n_top_hvg= n_top_hvg,
        use_hvg = use_hvg,
        target=target,
        max_spots=max_spots,
        min_spots=min_spots,
        sliding=sliding,
        stride=stride,
        min_genes=min_genes,
        min_cells=min_cells,
    )

    if use_condition is None:
        use_condition = metadata["n_conditions"] > 1

    return IDEANModel(
        adatas=adatas,
        train_set=train_set,
        eval_set=eval_set,
        gene_names=gene_names,
        offsets=offsets,
        metadata=metadata,
        use_condition=bool(use_condition),
        use_domain_adversarial=use_domain_adversarial,
        domain_adv_weight=domain_adv_weight,
        domain_adv_gamma=domain_adv_gamma,
        domain_label_smoothing=domain_label_smoothing,
        lr_domain=lr_domain,

        hidden_size=hidden_size,
        latent_size=latent_size,
        
        dropout=dropout,
        var_eps=var_eps,
        max_cluster=max_cluster,
        knn_k=knn_k,
        spatial_weight=spatial_weight,
        radius_scale=radius_scale,
        lr_encoder=lr_encoder,
        lr_decoder=lr_decoder,
        lr_classifier=lr_classifier,
        classifier_weight=classifier_weight,
        seed=seed,
        use_gpu=use_gpu,
    )
