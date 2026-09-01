import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import normalize
from . import data_downsample_new_1
import logging
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import numba as nb
from numba import jit
import collections
import random
from .data_downsample_new import downsample_cell,downsample_matrix_by_cell
from .data_augmentation_new import random_augment,random_augmentation_cell
#%%
# 从均匀分布中获取每个spot采样的细胞数和细胞类型数
def get_param_from_uniform(num_sample,cells_min=None,cells_max=None,clusters_min=None,clusters_max=None):

    cell_count = np.asarray(np.ceil(np.random.uniform(int(cells_min),int(cells_max),size=num_sample)),dtype=int)
    cluster_count = np.asarray(np.ceil(np.clip(np.random.uniform(clusters_min,clusters_max,size=num_sample),1,cell_count)),dtype=int)
    return cell_count, cluster_count

# 从高斯分布中获取每个spot采样的细胞数和细胞类型数
def get_param_from_gaussian(num_sample,cells_min=None,cells_max=None,cells_mean=None,cells_std=None,clusters_mean=None,clusters_std=None):

    cell_count = np.asarray(np.ceil(np.clip(np.random.normal(cells_mean,cells_std,size=num_sample),int(cells_min),int(cells_max))),dtype=int)
    cluster_count = np.asarray(np.ceil(np.clip(np.random.normal(clusters_mean,clusters_std,size=num_sample),1,cell_count)),dtype=int)
    return cell_count,cluster_count

# 从用空间数据估计的cell counts中获取每个spot采样的细胞数和细胞类型数
def get_param_from_cell_counts(
    num_sample,
    cell_counts,
    cluster_sample_mode='gaussian',
    cells_min=None,cells_max=None,
    cells_mean=None,cells_std=None,
    clusters_mean=None,clusters_std=None,
    clusters_min=None,clusters_max=None
):
    cell_count = np.asarray(np.ceil(np.clip(np.random.normal(cells_mean,cells_std,size=num_sample),int(cells_min),int(cells_max))),dtype=int)
    if cluster_sample_mode == 'gaussian':
        cluster_count = np.asarray(np.ceil(np.clip(np.random.normal(clusters_mean,clusters_std,size=num_sample),1,cell_count)),dtype=int)
    elif cluster_sample_mode == 'uniform':
        cluster_count = np.asarray(np.ceil(np.clip(np.random.uniform(clusters_min,clusters_max,size=num_sample),1,cell_count)),dtype=int)
    else:
        raise TypeError('Not correct sample method.')
    return cell_count,cluster_count
def init_sample_prob(sc_ad,celltype_key):
    print('### Initializing sample probability')
    #把细胞类型转换为数字表示，存储格式是DataFrame
    sc_ad.uns['celltype2num'] = pd.DataFrame(
        np.arange(len(sc_ad.obs[celltype_key].value_counts())).T, #[0,...,,120]
        index=sc_ad.obs[celltype_key].value_counts().index.values, #类型名
        columns=['celltype_num'] 
    )
    #把细胞类型的编号给obs的每个细胞？
    sc_ad.obs['celltype_num'] = [sc_ad.uns['celltype2num'].loc[c,'celltype_num'] for c in sc_ad.obs[celltype_key]]
    #sc_ad.obs['celltype_num'].value_counts() 每个类型数量   按照每种类型的数量计算类型概率，数量多的类型概率大
    cluster_p_unbalance = sc_ad.obs['celltype_num'].value_counts()/sc_ad.obs['celltype_num'].value_counts().sum()
    # 先开方再求取各个类型的概率
    cluster_p_sqrt = np.sqrt(sc_ad.obs['celltype_num'].value_counts())/np.sqrt(sc_ad.obs['celltype_num'].value_counts()).sum()
    #平均取，1/120？
    cluster_p_balance = pd.Series(
        np.ones(len(sc_ad.obs['celltype_num'].value_counts()))/len(sc_ad.obs['celltype_num'].value_counts()), 
        index=sc_ad.obs['celltype_num'].value_counts().index
    )
#     cluster_p_balance = np.ones(len(sc_ad.obs['celltype_num'].value_counts()))/len(sc_ad.obs['celltype_num'].value_counts())
    
    #求采到每个细胞的概率，1/cluster_p_unbalance[c] 是先根据类型赋予一个概率，对应类型数量越多的细胞，对单个细胞来说被采样概率越小？
    cell_p_balanced = [1/cluster_p_unbalance[c] for c in sc_ad.obs['celltype_num']]
    #归一化到和为1
    cell_p_balanced = np.array(cell_p_balanced)/np.array(cell_p_balanced).sum()
    
    sc_ad.obs['cell_p_balanced'] = cell_p_balanced
    sc_ad.uns['cluster_p_balance'] = cluster_p_balance
    sc_ad.uns['cluster_p_sqrt'] = cluster_p_sqrt
    sc_ad.uns['cluster_p_unbalance'] = cluster_p_unbalance
    return sc_ad


# 对某个axis调用numpy函数(numba版本)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

# 对某个axis计算均值(numba版本)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

# 对某个axis计算加和(numba版本)
def np_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)



# 获取每个cluster的采样概率
def get_cluster_sample_prob(sc_ad,mode):
    if mode == 'unbalance':
        cluster_p = sc_ad.uns['cluster_p_unbalance'].values
    elif mode == 'balance':
        cluster_p = sc_ad.uns['cluster_p_balance'].values
    elif mode == 'sqrt':
        cluster_p = sc_ad.uns['cluster_p_sqrt'].values
    else:
        raise TypeError('Balance argument must be one of [ None, banlance, sqrt ].')
    return cluster_p

def sample_cell(param_list,cluster_p,clusters,cluster_id,sample_exp,sample_cluster,cell_p_balanced,downsample_fraction=None,data_augmentation=True,max_rate=0.8,max_val=0.8,kth=0.2):
    exp = np.empty((len(param_list), sample_exp.shape[1]),dtype=np.float32) #(3333,2220)
    density = np.empty((len(param_list), sample_cluster.shape[1]),dtype=np.float32) #(3333,120)
    # detail = np.zeros((len(param_list), sample_cluster.shape[1], sample_exp.shape[1]),dtype=np.float32)
    for i in nb.prange(len(param_list)):
        params = param_list[i]
        num_cell = params[0]
        num_cluster = params[1]
        # np.searchsorted  在数组a中插入数组v（并不执行插入操作），返回一个下标列表，这个列表指明了v中对应元素应该插入在a中那个位置上
        # np.cumsum 计算轴向的累加和
        # np.random.rand 随机生成num_cluster个数； np.cumsum(cluster_p)：概率累加是1； np.searchsorted：根据索引判断采样的类型
        used_clusters = clusters[np.searchsorted(np.cumsum(cluster_p), np.random.rand(num_cluster), side="right")]
        cluster_mask = np.array([False]*len(cluster_id)) #(47429,) 全为的false 矩阵
        for c in used_clusters:
            cluster_mask = (cluster_id==c)|(cluster_mask)  #取样到的设置为true
        # print('cluster_mask',cluster_mask)
        # print('used_clusters',used_clusters)
        used_cell_ind = np.where(cluster_mask)[0] #当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
        used_cell_p = cell_p_balanced[cluster_mask] #各细胞被采到的概率
        used_cell_p = used_cell_p/used_cell_p.sum() # 归一化
        #相似的套路采样细胞
        sampled_cells = used_cell_ind[np.searchsorted(np.cumsum(used_cell_p), np.random.rand(num_cell), side="right")]
        
        # for j in  np.unique(cluster_id[sampled_cells]):
  
        #     detail[i,j,:] = np_mean(sample_exp[sampled_cells[cluster_id[sampled_cells]==j],:],axis=0).astype(np.float32)
            
        
        combined_exp = np_sum(sample_exp[sampled_cells,:],axis=0).astype(np.float32)
        # detail[i] = sampled_cells
        if data_augmentation:
            # print('1')
            combined_exp = random_augmentation_cell(combined_exp,max_rate=max_rate,max_val=max_val,kth=kth)
        if downsample_fraction is not None:
            combined_exp = downsample_cell(combined_exp, downsample_fraction)
        combined_clusters = np_sum(sample_cluster[cluster_id[sampled_cells]],axis=0).astype(np.float32)
        exp[i,:] = combined_exp
        density[i,:] = combined_clusters
    return exp,density

def generate_sample_array(sc_ad, used_genes):
    if used_genes is not None:
        sc_df = sc_ad.to_df().loc[:,used_genes]
    else:
        sc_df = sc_ad.to_df()
    return sc_df.values

def sample_cell_from_clusters(cluster_sample_list,ncell_sample_list,cluster_id,sample_exp,sample_cluster,cell_p_balanced,downsample_fraction=None,data_augmentation=True,max_rate=0.8,max_val=0.8,kth=0.2):
    exp = np.empty((len(cluster_sample_list), sample_exp.shape[1]),dtype=np.float32)
    density = np.empty((len(cluster_sample_list), sample_cluster.shape[1]),dtype=np.float32)
    for i in nb.prange(len(cluster_sample_list)):
        used_clusters = np.where(cluster_sample_list[i] == 1)[0]
        num_cell = ncell_sample_list[i]
        cluster_mask = np.array([False]*len(cluster_id))
        for c in used_clusters:
            cluster_mask = (cluster_id==c)|(cluster_mask)
        used_cell_ind = np.where(cluster_mask)[0]
        used_cell_p = cell_p_balanced[cluster_mask]
        used_cell_p = used_cell_p/used_cell_p.sum()
        sampled_cells = used_cell_ind[np.searchsorted(np.cumsum(used_cell_p), np.random.rand(num_cell), side="right")]
        combined_exp = np_sum(sample_exp[sampled_cells,:],axis=0).astype(np.float32)
        if data_augmentation:
            # print('1')
            combined_exp = random_augmentation_cell(combined_exp,max_rate=max_rate,max_val=max_val,kth=kth)
        if downsample_fraction is not None:
            combined_exp = downsample_cell(combined_exp, downsample_fraction)
        combined_clusters = np_sum(sample_cluster[cluster_id[sampled_cells]],axis=0).astype(np.float32)
        exp[i,:] = combined_exp
        density[i,:] = combined_clusters
    return exp,density

def generate_sm_adata_new(
    sc_ad,
    celltype_key,
    num_sample: int, 
    used_genes=None,
    balance_mode=['unbalance','sqrt','balance'],
    cell_sample_method='gaussian',
    cluster_sample_method='gaussian',
    cell_counts=None,
    downsample_fraction=None,
    data_augmentation=True,
    max_rate=0.8,max_val=0.8,kth=0.2,
    cells_min=1,cells_max=20,
    cells_mean=10,cells_std=5,
    clusters_mean=None,clusters_std=None,
    clusters_min=None,clusters_max=None,
    cell_sample_counts=None,cluster_sample_counts=None,
    ncell_sample_list=None,
    cluster_sample_list=None,
):
    if not 'cluster_p_unbalance' in sc_ad.uns:
        sc_ad = init_sample_prob(sc_ad,celltype_key)
    num_sample_per_mode = num_sample//len(balance_mode) #10000/3 计算每种采样模式的数量
    cluster_ordered = np.array(sc_ad.obs['celltype_num'].value_counts().index) #[0,...,119]
    cluster_num = len(cluster_ordered) #120
    cluster_id = sc_ad.obs['celltype_num'].values #(47429,) 每个细胞相应的类型（数字表示）
    cluster_mask = np.eye(cluster_num) #(120,120)
    
    if (cell_sample_counts is None) or (cluster_sample_counts is None):
        if cell_counts is not None:
            cells_mean = np.mean(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
            cells_std = np.std(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
            cells_min = int(np.min(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
            cells_max = int(np.max(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
        if clusters_mean is None:
            clusters_mean = cells_mean/2
        if clusters_std is None:
            clusters_std = cells_std/2
        if clusters_min is None:
            clusters_min = cells_min
        if clusters_max is None:
            clusters_max = np.min((cells_max//2,cluster_num))

        if cell_counts is not None:
            cell_sample_counts, cluster_sample_counts = get_param_from_cell_counts(num_sample_per_mode,cell_counts,cluster_sample_method,cells_mean=cells_mean,cells_std=cells_std,cells_max=cells_max,cells_min=cells_min,clusters_mean=clusters_mean,clusters_std=clusters_std,clusters_min=clusters_min,clusters_max=clusters_max)
        elif cell_sample_method == 'gaussian':
            cell_sample_counts, cluster_sample_counts = get_param_from_gaussian(num_sample_per_mode,cells_mean=cells_mean,cells_std=cells_std,cells_max=cells_max,cells_min=cells_min,clusters_mean=clusters_mean,clusters_std=clusters_std)
        elif cell_sample_method == 'uniform':
            cell_sample_counts, cluster_sample_counts = get_param_from_uniform(num_sample_per_mode,cells_max=cells_max,cells_min=cells_min,clusters_min=clusters_min,clusters_max=clusters_max)
        else:
            raise TypeError('Not correct sample method.')
            
    if cluster_sample_list is None or ncell_sample_list is None:
        params = np.array(list(zip(cell_sample_counts, cluster_sample_counts)))

        sample_data_list = []
        sample_labels_list = []
        for b in balance_mode:
            print(f'### Genetating simulated spatial data using scRNA data with mode: {b}')
            cluster_p = get_cluster_sample_prob(sc_ad,b)  #各类型的概率值
            if downsample_fraction is not None:
                if downsample_fraction > 0.035:
                    sample_data,sample_labels = sample_cell(
                        param_list=params,
                        cluster_p=cluster_p,
                        clusters=cluster_ordered,
                        cluster_id=cluster_id,
                        sample_exp=generate_sample_array(sc_ad,used_genes),
                        sample_cluster=cluster_mask,
                        cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                        downsample_fraction=downsample_fraction,
                        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                    )
                else:
                    sample_data,sample_labels = sample_cell(
                        param_list=params,
                        cluster_p=cluster_p,
                        clusters=cluster_ordered,
                        cluster_id=cluster_id,
                        sample_exp=generate_sample_array(sc_ad,used_genes),
                        sample_cluster=cluster_mask,
                        cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                    )
                    # logging.warning('### Downsample data with python backend')
                    sample_data = downsample_matrix_by_cell(sample_data, downsample_fraction, numba_end=False)
            else:
                sample_data,sample_labels = sample_cell(
                    param_list=params,
                    cluster_p=cluster_p,
                    clusters=cluster_ordered,
                    cluster_id=cluster_id,
                    sample_exp=generate_sample_array(sc_ad,used_genes),
                    sample_cluster=cluster_mask,
                    cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                    data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                )
    #         if data_augmentation:
    #             sample_data = random_augment(sample_data)
            sample_data_list.append(sample_data)
            sample_labels_list.append(sample_labels)
    else:
        sample_data_list = []
        sample_labels_list = []
        for b in balance_mode:
            print(f'### Genetating simulated spatial data using scRNA data with mode: {b}')
            cluster_p = get_cluster_sample_prob(sc_ad,b)
            if downsample_fraction is not None:
                if downsample_fraction > 0.035:
                    sample_data,sample_labels = sample_cell_from_clusters(
                        cluster_sample_list=cluster_sample_list,
                        ncell_sample_list=ncell_sample_list,
                        cluster_id=cluster_id,
                        sample_exp=generate_sample_array(sc_ad,used_genes),
                        sample_cluster=cluster_mask,
                        cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                        downsample_fraction=downsample_fraction,
                        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                    )
                else:
                    sample_data,sample_labels = sample_cell_from_clusters(
                        cluster_sample_list=cluster_sample_list,
                        ncell_sample_list=ncell_sample_list,
                        cluster_id=cluster_id,
                        sample_exp=generate_sample_array(sc_ad,used_genes),
                        sample_cluster=cluster_mask,
                        cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                    )
                    # logging.warning('### Downsample data with python backend')
                    sample_data = downsample_matrix_by_cell(sample_data, downsample_fraction, numba_end=False)
            else:
                sample_data,sample_labels = sample_cell_from_clusters(
                    cluster_sample_list=cluster_sample_list,
                    ncell_sample_list=ncell_sample_list,
                    cluster_id=cluster_id,
                    sample_exp=generate_sample_array(sc_ad,used_genes),
                    sample_cluster=cluster_mask,
                    cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                    data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                )
            sample_data_list.append(sample_data)
            sample_labels_list.append(sample_labels)
            
    sm_data_1 = np.concatenate(sample_data_list)
    sm_labels = np.concatenate(sample_labels_list)
    sm_data_mtx = csr_matrix(sm_data_1)
    sm_ad = anndata.AnnData(sm_data_mtx)
    sm_ad.var.index = sc_ad.var_names
            
    sm_labels_p = (sm_labels.T/sm_labels.sum(axis=1)).T
            
    sm_ad.obsm['label'] = pd.DataFrame(sm_labels_p,columns=np.array(sc_ad.obs[celltype_key].value_counts().index.values),index=sm_ad.obs_names)         
    return sm_ad
