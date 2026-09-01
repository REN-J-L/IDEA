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
import scipy.sparse as sp
def downsample_sm_spot_counts(sm_ad,st_ad):
    fitdistrplus = importr('fitdistrplus') #fitdistrplus 常用于分布拟合，比如拟合数据到正态分布、对数正态分布、伽马分布
    lib_sizes = robjects.FloatVector(np.array(st_ad.X.sum(1)).reshape(-1)) #将 Python 中的 st_ad.X.sum(1) 结果转换为 R 可以处理的向量类型 FloatVector
    res = fitdistrplus.fitdist(lib_sizes,'lnorm') #指定目标分布为对数正态分布
    loc = res[0][0] #分布的均值参数
    scale = res[0][1] #分布的标准差参数
    # sm_mtx_count = sm_ad.X.toarray() #取sm数据
        
    if sp.issparse(sm_ad.X):
        sm_mtx_count = sm_ad.X.toarray()
    else:
        sm_mtx_count = sm_ad.X
    sample_cell_counts = np.random.lognormal(loc,scale,sm_ad.shape[0]) #生成与st数据相同分别的总counts
    sm_mtx_count_index = np.argsort(np.argsort(sm_mtx_count.sum(1))) #argsort()是将X中的元素从小到大排序后，提取对应的索引index
    sample_cell_counts_sort = np.sort(sample_cell_counts) #将生成的总counts再排序
    sm_mtx_count_sortalign = sample_cell_counts_sort[sm_mtx_count_index]
    st_ad_X = st_ad.X#.A
    st_ad_P = st_ad_X/st_ad_X.sum(1).reshape(st_ad_X.shape[0],1)
    st_ad_P_MAX = st_ad_P.max()
    # sm_mtx_count_lb = downsample_matrix_by_cell(st_ad_P_MAX,sm_mtx_count,sm_mtx_count_sortalign.astype(np.int64), numba_end=False)
    sm_mtx_count_lb_1 = downsample_matrix_by_cell(st_ad_P_MAX,sm_mtx_count,sample_cell_counts.astype(np.int64), numba_end=False)
    # sm_ad.uns['down_sample'] = sm_mtx_count_lb
    sm_ad.uns['down_sample_1'] = sm_mtx_count_lb_1

# Cite from https://github.com/numba/numba-examples
#@numba.jit(nopython=True, parallel=True)
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
#@numba.jit(nopython=True, parallel=False)
def compute_bin(x, bin_edges):   #readsGet,cumCounts
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_max = bin_edges[-1]
    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin
    bin = np.searchsorted(bin_edges, x)-1 #在数组a中插入数组v（并不执行插入操作），返回一个下标列表，这个列表指明了v中对应元素应该插入在a中那个位置上。
    if bin < 0 or bin >= n:
        return None
    else:
        return bin

# Modified from https://github.com/numba/numba-examples
#@numba.jit(nopython=True, parallel=False)
def numba_histogram(a, bin_edges): ##readsGet,cumCounts
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
    n = np.floor(np.sum(cell_counts) * fraction) #np.floor 向下取整
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

def downsample_per_cell_python(c,per_cell_counts,st_ad_P_MAX):
    cell_counts,new_cell_counts = c,per_cell_counts#param[0],param[1]
    n = new_cell_counts
    # index = np.where(cell_counts/cell_counts.sum()>st_ad_P_MAX)
    # cell_counts[index] = (cell_counts.sum()-cell_counts[index])*st_ad_P_MAX/(1-st_ad_P_MAX)
    if n < np.sum(cell_counts):
        readsGet = np.sort(random.sample(range(np.intp(np.sum(cell_counts))), np.intp(n))) #从范围 [0, 总计数-1] 中随机选择 n 个位置（不重复）
        cumCounts = np.concatenate((np.array([0]),np.cumsum(cell_counts)))
        counts_new = numba_histogram(readsGet,cumCounts)[0]
        counts_new = counts_new.astype(np.float32)
        return counts_new
    else:
        return cell_counts.astype(np.float32)

def downsample_matrix_by_cell(st_ad_P_MAX, matrix,per_cell_counts,numba_end=True):
    if numba_end:
        downsample_func = downsample_per_cell
    else:
        downsample_func = downsample_per_cell_python
    matrix_ds = [downsample_func(c,per_cell_counts[i],st_ad_P_MAX) for i,c in enumerate(matrix)]
    return np.array(matrix_ds)

# ps. slow speed.
def downsample_matrix_total(matrix,fraction):
    matrix_flat = matrix.reshape(-1)
    matrix_flat_ds = downsample_cell(matrix_flat,fraction)
    matrix_ds = matrix_flat_ds.reshape(matrix.shape)
    return matrix_ds

