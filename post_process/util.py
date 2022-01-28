import numpy as np
from tqdm import tqdm
import os
import shutil
import time
import torch

import scipy
import math

def db_augmentation(query_vecs, reference_vecs, sim_mat, top_k=10):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    print("DBAing...")
    start_time = time.time()
    query_vecs, reference_vecs = np.array(query_vecs), np.array(reference_vecs)

    weights = np.logspace(0, -2., top_k+1)

    # Query augmentation
    _,indices = sim_mat.topk(12, largest=False)

    top_k_ref = reference_vecs[indices[:, :top_k], :]
    query_vecs = np.tensordot(weights, np.concatenate([np.expand_dims(query_vecs, 1), top_k_ref], axis=1), axes=(0, 1))
    # Reference augmentation
    if type(query_vecs) is np.ndarray:
        query_vecs = torch.from_numpy(query_vecs)

    print('=> DBA in {:.2f} seconds'.format(time.time() - start_time))
    return query_vecs

def average_query_expansion(query, gallery, dis, top_k=1):
    """
    Average Query Expansion (AQE)
    Ondrej Chum, et al. "Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval,"
    International Conference of Computer Vision. 2007.
    https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
    """
    print("AQEing...")
    start_time = time.time()
    if type(dis) is np.ndarray:
        dis = torch.from_numpy(dis)

    _,sorted_index = dis.topk(4, largest=False)
    sorted_index = sorted_index[:, :top_k]
    sorted_index = sorted_index.reshape(-1)
    requery = gallery[sorted_index].view(query.shape[0], -1, query.shape[1]).sum(dim=1)
    requery = (requery + query)/ (top_k + 1 )
    query = requery
    print('=> AQE in {:.2f} seconds'.format(time.time() - start_time))
    return query

def average_query_expansion_gallery(query, gallery,dis, top_k=1):
    # # Reference augmentation

    print("AQEing in gallery...")
    start_time = time.time()
    if type(dis) is np.ndarray:
        dis = torch.from_numpy(dis)

    _,sorted_index = dis.topk(4, largest=False)
    sorted_index = sorted_index[:, 1:top_k+1]

    sorted_index = sorted_index.reshape(-1)
    regallery = gallery[sorted_index].view(gallery.shape[0], -1, gallery.shape[1]).sum(dim=1)
    regallery = (regallery + gallery)/ (top_k + 1 )
    print('=> AQE in {:.2f} seconds'.format(time.time() - start_time))
    return regallery


def fast_kr_sparse(q_g_dist, q_q_dist, g_g_dist, k1=15, k2=3, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    print("fast Reranking")
    start_time = time.time()
    end_time = time.time()

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    g_q_dist = q_g_dist.t().clone()

    topk = max(k1 + 1, k2)

    all_initial_rank_list = []
    batch_size = 128
    original_dist_batch = torch.zeros(batch_size, len(q_q_dist)+len(g_g_dist))
    total_num = math.ceil(len(q_q_dist) / float(batch_size))
    for i in tqdm(range(total_num)):
        left_id = i * batch_size
        right_id = min((i+1) * batch_size, len(q_q_dist))
        batch_len = right_id - left_id
        original_dist_batch_tmp = original_dist_batch[:batch_len]
        original_dist_batch_tmp[:, :len(q_q_dist)] = q_q_dist[left_id:right_id]
        original_dist_batch_tmp[:, len(q_q_dist):] = q_g_dist[left_id:right_id]
        original_dist_batch_tmp = original_dist_batch_tmp ** 2
        m, _ = original_dist_batch_tmp.max(1)
        original_dist_batch_tmp = original_dist_batch_tmp / m.reshape(-1, 1)
        q_q_dist[left_id:right_id] = original_dist_batch_tmp[:, :len(q_q_dist)]
        q_g_dist[left_id:right_id] = original_dist_batch_tmp[:, len(q_q_dist):]
        _, initial_rank = original_dist_batch_tmp.topk(topk, largest=False)
        all_initial_rank_list.append(initial_rank)

    total_num = math.ceil(len(g_g_dist) / float(batch_size))
    for i in tqdm(range(total_num)):
        left_id = i * batch_size
        right_id = min((i+1) * batch_size, len(g_g_dist))
        batch_len = right_id - left_id
        original_dist_batch_tmp = original_dist_batch[:batch_len]
        original_dist_batch_tmp[:, :len(q_q_dist)] = g_q_dist[left_id:right_id]
        original_dist_batch_tmp[:, len(q_q_dist):] = g_g_dist[left_id:right_id]
        original_dist_batch_tmp = original_dist_batch_tmp ** 2
        m, _ = original_dist_batch_tmp.max(1)
        original_dist_batch_tmp = original_dist_batch_tmp / m.reshape(-1, 1)
        g_q_dist[left_id:right_id] = original_dist_batch_tmp[:, :len(q_q_dist)]
        g_g_dist[left_id:right_id] = original_dist_batch_tmp[:, len(q_q_dist):]
        _, initial_rank = original_dist_batch_tmp.topk(topk, largest=False)
        all_initial_rank_list.append(initial_rank)

    initial_rank = torch.cat(all_initial_rank_list)
    initial_rank = initial_rank.numpy()

    V = scipy.sparse.lil_matrix((all_num, all_num))
    print('prepare time: {:.2f}'.format(time.time() - end_time))

    original_dist_for_1 = np.zeros(gallery_num)
    for i in tqdm(range(all_num)):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        if i < len(q_q_dist):
            original_dist_for_1[:len(q_q_dist)] = q_q_dist[i]
            original_dist_for_1[len(q_q_dist):] = q_g_dist[i]
        else:
            j = i - len(q_q_dist)
            original_dist_for_1[:len(q_q_dist)] = g_q_dist[j]
            original_dist_for_1[len(q_q_dist):] = g_g_dist[j]

        weight = np.exp(-original_dist_for_1[k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    end_time = time.time()
    if k2 != 1:
        X_list = []
        Y_list = []
        V_list = []
        array_buffer = np.zeros((k2, all_num))
        for i in tqdm(range(all_num)):
            V[initial_rank[i,:k2]].toarray(out=array_buffer)
            m = array_buffer.mean(axis=0)
            y = np.where(m)[0]
            x = np.zeros_like(y)
            x[:] = i
            v = m[y]
            X_list.append(x)
            Y_list.append(y)
            V_list.append(v)
        X = np.concatenate(X_list)
        Y = np.concatenate(Y_list)
        V = np.concatenate(V_list)
        V = scipy.sparse.csr_matrix((V, (X, Y)), shape=(all_num, all_num)).tolil()
    del initial_rank
    print('qe time: {:.2f}'.format(time.time() - end_time))

    V_t = V.transpose()
    invIndex = []
    for i in tqdm(range(gallery_num)):
        invIndex.append(V_t[i].nonzero()[1])

    jaccard_dist = np.zeros((len(q_q_dist), len(g_g_dist)),dtype = np.float32)

    temp_min = np.zeros(gallery_num,dtype=np.float32)
    for i in tqdm(range(query_num)):
        temp_min[:] = 0
        indNonZero = V[i].nonzero()[1]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            t1 = V[i,indNonZero[j]]
            t2 = V[indImages[j],indNonZero[j]].toarray().reshape(-1)
            temp_min[indImages[j]] = temp_min[indImages[j]]+ np.minimum(t1, t2)
        temp_min = 1-temp_min/(2.-temp_min)
        jaccard_dist[i] = temp_min[len(q_q_dist):]

    new_q_g = jaccard_dist*(1-lambda_value) + q_g_dist.numpy()*lambda_value
    del V
    del jaccard_dist
    print('=> k-reciprocal in {:.2f} seconds'.format(time.time() - start_time))
    return torch.from_numpy(new_q_g)
