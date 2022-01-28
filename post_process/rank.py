import argparse
import os
import shutil
from PIL import Image, ImageDraw
import numpy as np
import torch
from tqdm import tqdm
import sklearn.decomposition
import time
import math

from util import  db_augmentation, average_query_expansion,average_query_expansion_gallery
from util import fast_kr_sparse

parser = argparse.ArgumentParser(description='Post process for retrieval')
parser.add_argument('feature', metavar='DIR', nargs='*',
                    help='path to feature')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')


def cat_feature(path):
    names = os.listdir(path)
    names.sort()
    features = []
    for vectors_name in tqdm(names):
        vectors = torch.load(os.path.join(path, vectors_name)).reshape(1,-1)
        vectors = torch.nn.functional.normalize(vectors)
        features.append(vectors)
    features = torch.cat(features, dim=0).cpu()
    return features, names

# Decomposition
def decomposition_feature(query, gallery, k=512):
    print("PCAing...")
    start_time = time.time()
    pca = sklearn.decomposition.PCA(k)
    pca.fit(query)
    query = pca.transform(query)
    gallery = pca.transform(gallery)
    query = torch.tensor(query)
    gallery = torch.tensor(gallery)
    print("=> finish PCA in {:.2f} seconds".format(time.time() - start_time))
    return query, gallery

#   Load query and gallery feature
def gain_features(feature_name_list):
    query_feature_list = []
    gallery_feature_list = []

    query_name_list = [] 
    gallery_name_list = []
    for path in feature_name_list:
        print('=> load feature: {}'.format(path))
        query_path = os.path.join(path, 'query_feat')
        query_feature, query_name = cat_feature(query_path)
        gallery_path = os.path.join(path, 'gallery_feat')
        gallery_feature, gallery_name = cat_feature(gallery_path)

        if gallery_feature.shape[1] == 12288:
            query_feature, gallery_feature = decomposition_feature(query_feature, gallery_feature, 6144)

        query_feature_list.append(query_feature)
        gallery_feature_list.append(gallery_feature)

        print(query_feature.shape)
        print(gallery_feature.shape)
        query_name_list.append(query_name)
        gallery_name_list.append(gallery_name)

    query_feature = torch.cat(query_feature_list, dim=1)
    gallery_feature = torch.cat(gallery_feature_list, dim=1)

    query_name_list_1 = query_name_list[0]
    gallery_name_list_1 = gallery_name_list[0]
    for idx in range(len(query_name_list)):
        assert query_name_list_1 == query_name_list[idx] and gallery_name_list_1 == gallery_name_list[idx]
    return query_feature, gallery_feature, query_name_list_1, gallery_name_list_1


def fast_gain_distance(query_feature, gallery_feature, cuda_index = 7):
    start_time = time.time()
    print('=> fast gain distance...')
    def get_dist_matrix_by_batch(query_feature, gallery_feature):
        batch_size = 128
        query_loader = torch.chunk(query_feature, math.ceil(len(query_feature) / batch_size))
        ture_batch_size = len(query_loader[0])
        q_g_dist_matrix_cpu = torch.zeros(query_feature.shape[0], gallery_feature.shape[0])
        for i, query in tqdm(enumerate(query_loader)):
            dist_matrix = torch.cdist(query, gallery_feature)
            q_g_dist_matrix_cpu[i*ture_batch_size: (i+1)*ture_batch_size] = dist_matrix.cpu()
        return q_g_dist_matrix_cpu

    query_feature = query_feature.cuda(cuda_index)
    gallery_feature = gallery_feature.cuda(cuda_index)
    end_time = time.time()
    q_g_dist_matrix_cpu = get_dist_matrix_by_batch(query_feature, gallery_feature)
    print('=> q_g time: {:.2f}. Shape: {}'.format(time.time() - end_time, q_g_dist_matrix_cpu.shape))

    end_time = time.time()
    q_q_dist_matrix_cpu = get_dist_matrix_by_batch(query_feature, query_feature)
    print('=> q_q time: {:.2f}. Shape: {}'.format(time.time() - end_time, q_q_dist_matrix_cpu.shape))
    
    end_time = time.time()
    g_g_dist_matrix_cpu = get_dist_matrix_by_batch(gallery_feature, gallery_feature)
    print('=> g_g time: {:.2f}. Shape: {}'.format(time.time() - end_time, g_g_dist_matrix_cpu.shape))

    #torch.cuda.empty_cache()

    q_g_dist_matrix_cpu = fast_kr_sparse(q_g_dist_matrix_cpu, q_q_dist_matrix_cpu, g_g_dist_matrix_cpu, k1=20, k2=6, lambda_value=0.3)
    print('=> gain distance in {:.2f} seconds'.format(time.time() - start_time))
    return q_g_dist_matrix_cpu, q_q_dist_matrix_cpu, g_g_dist_matrix_cpu

def gen_results(dist_matrix, query_name_list, gallery_name_list, rank=10):
    results = []
    idx = dist_matrix.topk(k=rank, dim=-1, largest=False)[1]
    with open('submission.csv', 'w') as f:
        for i in tqdm(range(len(query_name_list))):
            strs = query_name_list[i] + ".jpg,{" 
            gallery_index_i = idx[i]
            for index in gallery_index_i:
                strs += gallery_name_list[index] + ".jpg," 
            strs = strs[:-1]  + "}"
            print(strs)
            f.write(strs)
            f.write('\n')
            results.append(strs)
    return results

def gain_list(feature_list):
    for path in feature_list[:1]:
        quert_path = os.path.join(path, 'query_feat')
        query_vectors_list = os.listdir(quert_path)
        query_vectors_list.sort()
        gallery_path = os.path.join(path, 'gallery_feat')
        gallery_vectors_list = os.listdir(gallery_path)
        gallery_vectors_list.sort()
    return query_vectors_list, gallery_vectors_list


if __name__ == "__main__":
    args = parser.parse_args()

    '''
    feature_list = ["DIGIX_test_B_fishnet99_5153-2020-09-24_08:58:33", 
    "DIGIX_test_B_hrnet_w30_5308-2020-09-24_00:50:26",
    "DIGIX_test_B_hrnet_w18_5253-2020-09-28_10:28:21",
    "DIGIX_test_B_dla102x_5088-2020-09-24_01:45:08", 
    "DIGIX_test_B_resnet101_5059-2020-09-24_04:03:42"]
    '''

    query, gallery, query_name_list, gallery_name_list = gain_features(args.feature)
    query, gallery = decomposition_feature(query,gallery)
    
    # torch.save(query, "query.pt")
    # torch.save(gallery, "gallery.pt")
    # query = torch.load("query.pt")
    # gallery = torch.load( "gallery.pt")

    q_g, q_q, g_g = fast_gain_distance(query, gallery, cuda_index = args.gpu)


    query = average_query_expansion(query, gallery, q_g, top_k=1)
    query = db_augmentation(query, gallery, q_g, top_k=10)

    gallery = average_query_expansion_gallery(query, gallery, g_g, top_k=1)

    q_g, q_q, g_g = fast_gain_distance(query, gallery, cuda_index = args.gpu)
    # torch.save(q_g, "q_g.pt", pickle_protocol=4)
    # torch.save(q_q, "q_q.pt", pickle_protocol=4)
    # torch.save(g_g, "g_g.pt", pickle_protocol=4)

    # q_g = torch.load( "q_g.pt")
    query_name_list, gallery_name_list = gain_list(args.feature)
    gen_results(q_g, query_name_list, gallery_name_list)
