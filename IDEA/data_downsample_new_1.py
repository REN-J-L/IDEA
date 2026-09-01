import numba
import numpy as np
import multiprocessing as mp
from functools import partial
import random
import scanpy as sc
import anndata
from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import normalize
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from scipy.stats import truncnorm

def downsample_sm_spot_counts(sm_ad,st_ad):
    fitdistrplus = importr('fitdistrplus') #fitdistrplus 常用于分布拟合，比如拟合数据到正态分布、对数正态分布、伽马分布
    lib_sizes = robjects.FloatVector(np.array(st_ad.X.sum(1)).reshape(-1)) #将 Python 中的 st_ad.X.sum(1) 结果转换为 R 可以处理的向量类型 FloatVector

    res_lnorm = fitdistrplus.fitdist(lib_sizes,'lnorm') 
    loc_lnorm = res_lnorm[0][0] #分布的均值参数
    scale_lnorm = res_lnorm[0][1] #分布的标准差参数
    
    res_norm = fitdistrplus.fitdist(lib_sizes,'norm')
    loc_norm = res_norm[0][0] #分布的均值参数
    scale_norm = res_norm[0][1] #分布的标准差参数
    
    res_nbinom = fitdistrplus.fitdist(lib_sizes, 'nbinom')
    size_nbinom = res_nbinom[0][0]  # r
    prob_nbinom = res_nbinom[0][1]  # p
    
    aic_values = {
        'lnorm': res_lnorm.rx2('aic')[0],
        'norm': res_norm.rx2('aic')[0],
        'nbinom': res_nbinom.rx2('aic')[0]
    }
    bic_values = {
        'lnorm': res_lnorm.rx2('bic')[0],
        'norm': res_norm.rx2('bic')[0],
        'nbinom': res_nbinom.rx2('bic')[0]
    }
    
    combined_scores = {k: aic_values[k] + bic_values[k] for k in aic_values}
    best_dist = min(combined_scores, key=combined_scores.get)
    
    
    sm_mtx_count = sm_ad.X.toarray() #取sm数据
    
    if best_dist == 'lnorm':
        sample_cell_counts = np.random.lognormal(loc_lnorm,scale_lnorm,sm_ad.shape[0]) #生成与st数据相同分别的总counts
    elif best_dist == 'norm':
        a, b = (np.array(st_ad.X.sum(1)).min() - loc_norm)/scale_norm, (np.array(st_ad.X.sum(1)).max() - loc_norm)/scale_norm
        sample_cell_counts = truncnorm.rvs(a, b, loc=loc_norm, scale=scale_norm, size=sm_ad.shape[0])
    elif best_dist == 'nbinom':
        sample_cell_counts = res_nbinom.rvs(n=size_nbinom, p=prob_nbinom, size=sm_ad.shape[0])
        
    sm_mtx_count_lb = downsample_matrix_by_cell(sm_mtx_count,sample_cell_counts.astype(np.int64), numba_end=False)
    
    sm_ad.uns['down_sample'] = sm_mtx_count_lb

# Cite from https://github.com/numba/numba-examples
@numba.jit(nopython=True, parallel=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float32)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in numba.prange(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges

# Modified from https://github.com/numba/numba-examples
@numba.jit(nopython=True, parallel=False)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_max = bin_edges[-1]
    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin
    bin = np.searchsorted(bin_edges, x)-1
    if bin < 0 or bin >= n:
        return None
    else:
        return bin

# Modified from https://github.com/numba/numba-examples
@numba.jit(nopython=True, parallel=False)
def numba_histogram(a, bin_edges):
    hist = np.zeros((bin_edges.shape[0] - 1,), dtype=np.intp)
    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1
    return hist, bin_edges


# Modified from https://rdrr.io/bioc/scRecover/src/R/countsSampling.R
# Downsample cell reads to a fraction
@numba.jit(nopython=True, parallel=True)
def downsample_cell(cell_counts,fraction):
    n = np.floor(np.sum(cell_counts) * fraction)
    readsGet = np.sort(np.random.choice(np.arange(np.sum(cell_counts)), np.intp(n), replace=False))
    cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
    counts_new = numba_histogram(readsGet,cumCounts)[0]
    counts_new = counts_new.astype(np.float32)
    return counts_new

def downsample_cell_python(cell_counts,fraction):
    n = np.floor(np.sum(cell_counts) * fraction)
    readsGet = np.sort(random.sample(range(np.intp(np.sum(cell_counts))), np.intp(n)))
    cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
    counts_new = numba_histogram(readsGet,cumCounts)[0]
    counts_new = counts_new.astype(np.float32)
    return counts_new

@numba.jit(nopython=True, parallel=True)
def downsample_per_cell(cell_counts,new_cell_counts):
    n = new_cell_counts
    if n < np.sum(cell_counts):
        readsGet = np.sort(np.random.choice(np.arange(np.sum(cell_counts)), np.intp(n), replace=False))
        cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
        counts_new = numba_histogram(readsGet,cumCounts)[0]
        counts_new = counts_new.astype(np.float32)
        return counts_new
    else:
        return cell_counts.astype(np.float32)

def downsample_per_cell_python(param):
    cell_counts,new_cell_counts = param[0],param[1]
    n = new_cell_counts
    if n < np.sum(cell_counts):
        readsGet = np.sort(random.sample(range(np.intp(np.sum(cell_counts))), np.intp(n)))
        cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
        counts_new = numba_histogram(readsGet,cumCounts)[0]
        counts_new = counts_new.astype(np.float32)
        return counts_new
    else:
        return cell_counts.astype(np.float32)

def downsample_matrix_by_cell(matrix,per_cell_counts,n_cpus=None,numba_end=True):
    n_cpus=None
    
    if numba_end:
        downsample_func = downsample_per_cell
    else:
        downsample_func = downsample_per_cell_python
    if n_cpus is not None:
        with mp.Pool(n_cpus) as p:
            matrix_ds = p.map(downsample_func, zip(matrix,per_cell_counts))
    else:
        matrix_ds = [downsample_func((c,per_cell_counts[i])) for i,c in enumerate(matrix)]
    return np.array(matrix_ds)

# ps. slow speed.
def downsample_matrix_total(matrix,fraction):
    matrix_flat = matrix.reshape(-1)
    matrix_flat_ds = downsample_cell(matrix_flat,fraction)
    matrix_ds = matrix_flat_ds.reshape(matrix.shape)
    return matrix_ds