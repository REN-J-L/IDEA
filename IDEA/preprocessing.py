# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:38:02 2024

@author: 18271
"""

import scanpy as sc
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt



def filter_genes(adata, cell_count_cutoff=15, cell_percentage_cutoff2=0.05, nonz_mean_cutoff=1.12):
    r"""Plot the gene filter given a set of cutoffs and return resulting list of genes.

    Parameters
    ----------
    adata :
        anndata object with single cell / nucleus data.
    cell_count_cutoff :
        All genes detected in less than cell_count_cutoff cells will be excluded.
    cell_percentage_cutoff2 :
        All genes detected in at least this percentage of cells will be included.
    nonz_mean_cutoff :
        genes detected in the number of cells between the above mentioned cutoffs are selected
        only when their average expression in non-zero cells is above this cutoff.

    Returns
    -------
    a list of selected var_names
    """

    adata.var["n_cells"] = np.array((adata.X > 0).sum(0)).flatten()
    adata.var["nonz_mean"] = np.array(adata.X.sum(0)).flatten() / adata.var["n_cells"]

    cell_count_cutoff = np.log10(cell_count_cutoff)
    cell_count_cutoff2 = np.log10(adata.shape[0] * cell_percentage_cutoff2)
    nonz_mean_cutoff = np.log10(nonz_mean_cutoff)

    gene_selection = (np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff2)) | (
        np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff)
        & np.array(np.log10(adata.var["nonz_mean"]) > nonz_mean_cutoff)
    )
    gene_selection = adata.var_names[gene_selection]
    adata_shape = adata[:, gene_selection].shape

    fig, ax = plt.subplots()
    ax.hist2d(
        np.log10(adata.var["nonz_mean"]),
        np.log10(adata.var["n_cells"]),
        bins=100,
        norm=matplotlib.colors.LogNorm(),
        range=[[0, 0.5], [1, 4.5]],
    )
    ax.axvspan(0, nonz_mean_cutoff, ymin=0.0, ymax=(cell_count_cutoff2 - 1) / 3.5, color="darkorange", alpha=0.3)
    ax.axvspan(
        nonz_mean_cutoff,
        np.max(np.log10(adata.var["nonz_mean"])),
        ymin=0.0,
        ymax=(cell_count_cutoff - 1) / 3.5,
        color="darkorange",
        alpha=0.3,
    )
    plt.vlines(nonz_mean_cutoff, cell_count_cutoff, cell_count_cutoff2, color="darkorange")
    plt.hlines(cell_count_cutoff, nonz_mean_cutoff, 1, color="darkorange")
    plt.hlines(cell_count_cutoff2, 0, nonz_mean_cutoff, color="darkorange")
    plt.xlabel("Mean non-zero expression level of gene (log)")
    plt.ylabel("Number of cells expressing gene (log)")
    plt.title(f"Gene filter: {adata_shape[0]} cells x {adata_shape[1]} genes")
    plt.show()

    return gene_selection


#%%
def normalize_adata(ad,target_sum=None):
    ad_norm = sc.pp.normalize_total(ad,inplace=False,target_sum=1e4)
    #Normalize each cell by total counts (10000) over all genes, so that every cell has the same total count after normalization.
    ad_norm  = sc.pp.log1p(ad_norm['X'])
    #computes X = log(X+1)
    ad.layers['norm'] = ad_norm
    return ad

#%% qc
def qc(
    sc_ad, 
    st_ad,
    sc_filter_genes_min_cells = 1,
    st_filter_genes_min_cells = 1,
    sc_filter_cells_min_genes = 1,
    st_filter_cells_min_genes = 1,
    n_genes_by_counts_min = None, 
    n_genes_by_counts_max = None,
    pct_counts_mt_max = None,
    pct_counts_ribo_max = None,
    pct_counts_hb_max = None
):
    sc.pp.filter_genes(st_ad, min_cells = sc_filter_genes_min_cells)
    sc.pp.filter_genes(sc_ad, min_cells = st_filter_genes_min_cells)
    
    sc.pp.filter_cells(sc_ad, min_genes = sc_filter_cells_min_genes)
    sc.pp.filter_cells(st_ad, min_genes = st_filter_cells_min_genes)
    
    sc_ad.var_names = sc_ad.var_names.astype(str)
    sc_ad.var_names_make_unique()
    st_ad.var_names = st_ad.var_names.astype(str)
    st_ad.var_names_make_unique()


    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    sc_ad.var["mt"] = sc_ad.var_names.str.startswith("MT-")
    # ribosomal genes
    sc_ad.var["ribo"] = sc_ad.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    sc_ad.var["hb"] = sc_ad.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(
        sc_ad, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
        )
    sc.pl.violin(sc_ad, ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb'], jitter=0.4, multi_panel=True)

    if  n_genes_by_counts_min is not None:
        sc_ad = sc_ad[(sc_ad.obs['n_genes_by_counts'] > n_genes_by_counts_min)]
    if  n_genes_by_counts_max is not None:
        sc_ad = sc_ad[(sc_ad.obs['n_genes_by_counts'] < n_genes_by_counts_max)]
    if  pct_counts_mt_max is not None:
        sc_ad = sc_ad[(sc_ad.obs['pct_counts_mt'] < pct_counts_mt_max)]
    if  pct_counts_ribo_max is not None:
        sc_ad = sc_ad[(sc_ad.obs['pct_counts_ribo'] < pct_counts_ribo_max)]
    if  pct_counts_hb_max is not None:
        sc_ad = sc_ad[(sc_ad.obs['pct_counts_hb'] < pct_counts_hb_max)]
     
    return sc_ad, st_ad


#%%
def cohgenes(
    sc_ad, 
    st_ad, 
    celltype_key=None, 
    used_genes=None,
    sc_genes=None,
    st_genes=None,
    layer='norm',
    deg_method=None,
    log2fc_min=0.5, 
    pval_cutoff=0.01, 
    n_top_markers:int=200, 
    n_top_hvg=None,
    n_top_hvg_sc=None,
    pct_diff=None, 
    pct_min=0.1,
    only_use_common = False
):
    sc_ad = normalize_adata(sc_ad,target_sum=1e4)
    st_ad = normalize_adata(st_ad,target_sum=1e4)
    
    overlaped_genes = np.intersect1d(sc_ad.var_names,st_ad.var_names) 
    sc_ad = sc_ad[:,overlaped_genes].copy()  
    st_ad = st_ad[:,overlaped_genes].copy()
    
    if used_genes is None and only_use_common == False:
        if st_genes is None:
            if n_top_hvg is None:
                st_genes = st_ad.var_names  #如果没有指定使用空转数据的n_top_hvg,就用两个数据集重叠的基
            else:
                st_genes = find_st_hvg(st_ad, n_top_hvg) #如果指定了，就用scanpy找高变基因
        if sc_genes is None:
            if n_top_hvg_sc is not None: #如果没有指定使用sc数据的n_top_hvg,就用sc每个类的差异maker基因
                sc_genes = find_st_hvg(sc_ad, n_top_hvg)
            else:
                sc_ad = sc_ad[:, st_genes].copy() #提取相关数据
                sc_genes = find_sc_markers(sc_ad, celltype_key, layer, deg_method, log2fc_min, pval_cutoff, n_top_markers, pct_diff, pct_min)#计算单细胞亚群差异基因
        used_genes = np.intersect1d(sc_genes,st_genes) #再找交集
    else:
        used_genes = overlaped_genes
    
    sc_ad = sc_ad[:,used_genes].copy() # 提取数据
    st_ad = st_ad[:,used_genes].copy()
    sc.pp.filter_cells(sc_ad,min_genes=1) #质控
    sc.pp.filter_cells(st_ad,min_genes=1)
    print(f'### Used gene numbers: {len(used_genes)}')
    return sc_ad, st_ad

#%% find_st_hvg
def find_st_hvg(st_ad,n_top_hvg=None):
    print('### Finding HVG in spatial...')
    sc.pp.highly_variable_genes(st_ad,n_top_genes=n_top_hvg,flavor='seurat_v3')
    return st_ad.var_names[st_ad.var['highly_variable'] == True]
    
#%% find_sc_markers
# 计算单细胞亚群差异基因
def find_sc_markers(sc_ad, celltype_key, layer='norm', deg_method=None, log2fc_min=0.5, pval_cutoff=0.01, n_top_markers=200, pct_diff=None, pct_min=0.1):
    print('### Finding marker genes...')
    # filter celltype contain only one sample.
    filtered_celltypes = list(sc_ad.obs[celltype_key].value_counts()[(sc_ad.obs[celltype_key].value_counts() == 1).values].index)
    if len(filtered_celltypes) > 0:
        sc_ad = sc_ad[sc_ad.obs[~(sc_ad.obs[celltype_key].isin(filtered_celltypes))].index,:].copy()
        print(f'### Filter cluster contain only one sample: {filtered_celltypes}')

    sc.tl.rank_genes_groups(sc_ad, groupby=celltype_key, pts=True, layer=layer, use_raw=False, method=deg_method) #按celltype_key找差异表达基因
    marker_genes_dfs = []
    for c in np.unique(sc_ad.obs[celltype_key]):
        tmp_marker_gene_df = sc.get.rank_genes_groups_df(sc_ad, group=c, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min) #按celltype_key提取相关数据
        if (tmp_marker_gene_df.empty is not True):#如果非空
            tmp_marker_gene_df.index = tmp_marker_gene_df.names 
            tmp_marker_gene_df.loc[:,celltype_key] = c
            if pct_diff is not None: #
                pct_diff_genes = sc_ad.var_names[np.where((sc_ad.uns['rank_genes_groups']['pts'][c]-sc_ad.uns['rank_genes_groups']['pts_rest'][c]) > pct_diff)]
                #sc_ad.uns['rank_genes_groups']['pts'][c]: 这个表达式指的是第 c 个细胞群体中每个基因的表达比例
                #sc_ad.uns['rank_genes_groups']['pts_rest'][c]: 这个表达式是第 c 个细胞群体中每个基因在其他群体中的表达比例
                tmp_marker_gene_df = tmp_marker_gene_df.loc[np.intersect1d(pct_diff_genes, tmp_marker_gene_df.index),:]
            if pct_min is not None:
                # pct_min_genes = sc_ad.var_names[np.where((sc_ad.uns['rank_genes_groups']['pts'][c]) > pct_min)]
                tmp_marker_gene_df = tmp_marker_gene_df[tmp_marker_gene_df['pct_nz_group'] > pct_min] #非零表达比例
            if n_top_markers is not None:  #找top_n
                tmp_marker_gene_df = tmp_marker_gene_df.sort_values('logfoldchanges',ascending=False)
                tmp_marker_gene_df = tmp_marker_gene_df.iloc[:n_top_markers,:]
            marker_genes_dfs.append(tmp_marker_gene_df) 
    marker_gene_df = pd.concat(marker_genes_dfs,axis=0)  #
    print(marker_gene_df[celltype_key].value_counts())
    all_marker_genes = np.unique(marker_gene_df.names)
    return all_marker_genes

    
# #%%
# def make_dataset(
#     sc_ad,
#     st_ad,
        
# ):
    