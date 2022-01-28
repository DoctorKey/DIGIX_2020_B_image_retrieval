import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import sklearn.decomposition

from .. import losses, ramps, cli
from ..eval import eval_retrieval_by_labelfile
from ..architectures import create_model, load_pretrained
from ..model.CGD_margin_loss import CGD_margin_loss
from ..run_context import RunContext
from ..dataset.datasets import get_dataset_config
from ..dataset.dataloader import create_eval_loader
from ..utils import AverageMeterSet, parameters_string

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0

def main(context):
    global global_step
    global best_prec1
    global args

    args = context.args

    if context.vis_log is not None:
        context.vis_log.add_text('hparams', cli.arg2str(args))

    checkpoint_path = context.transient_dir

    dataset_config = get_dataset_config(args.dataset, args=args)
    query_dataset = dataset_config.get('query_dataset')
    gallery_dataset = dataset_config.get('gallery_dataset')
    extract_train = dataset_config.get('extract_train', None)
    
    num_classes = dataset_config.get('num_classes')
    label_file = dataset_config.get('label')

    query_loader = create_eval_loader(query_dataset, args=args)
    gallery_loader = create_eval_loader(gallery_dataset, args=args)

    LOG.info("=> creating model '{arch}'".format(arch=args.arch))
    backbone = create_model(args.arch, num_classes, args.pretext_class, detach_para=False, DataParallel=False)
    model = CGD_margin_loss(backbone, args.gd_config, args.feature_dim, num_classes, args).cuda()
    LOG.info(parameters_string(model))
    model = load_pretrained(model, args.pretrained, args.arch, LOG, DataParallel=True)


    cudnn.benchmark = True

    feat_dir = os.path.join(context.result_dir, 'query_feat')
    prop_dir = os.path.join(context.result_dir, 'query_prop')
    LOG.info("=> save query feature in {}".format(feat_dir))
    save_feature(query_loader, model, 0, LOG, args.print_freq, feat_dir, prop_dir)

    feat_dir = os.path.join(context.result_dir, 'gallery_feat')
    prop_dir = os.path.join(context.result_dir, 'gallery_prop')
    LOG.info("=> save gallery feature in {}".format(feat_dir))
    save_feature(gallery_loader, model, 0, LOG, args.print_freq, feat_dir, prop_dir)


    query_name_list, query = load_feat(os.path.join(context.result_dir, 'query_feat'))
    gallery_name_list, gallery = load_feat(os.path.join(context.result_dir, 'gallery_feat'))
    LOG.info('=> loaded feature. query: {} gallery: {}'.format(len(query_name_list), len(gallery_name_list)))

    query = F.normalize(query)
    gallery = F.normalize(gallery)
    #query, gallery = reduce_feat(query, gallery)

    # retrieval in query
    nn_idx = nn_search(query, None, args.print_freq).cpu()
    result_file = os.path.join(context.result_dir, 'submission_query.csv')
    gen_result_file(result_file, query_name_list, np.array(query_name_list), nn_idx)
    if label_file is not None:
        LOG.info('=> retrieval in query')
        eval_retrieval_by_labelfile(result_file, label_file, LOG)

    nn_idx = nn_search(query, gallery, args.print_freq).cpu()
    result_file = os.path.join(context.result_dir, 'submission.csv')
    gen_result_file(result_file, query_name_list, np.array(gallery_name_list), nn_idx)
    if label_file is not None:
        LOG.info('=> retrieval in gallery')
        eval_retrieval_by_labelfile(result_file, label_file, LOG)

    if extract_train is not None:
        extract_train_loader = create_eval_loader(extract_train, args=args)
        feat_dir = os.path.join(context.result_dir, 'train_feat')
        prop_dir = os.path.join(context.result_dir, 'train_prop')
        LOG.info("=> save train feature in {}".format(feat_dir))
        save_feature(extract_train_loader, model, 0, LOG, args.print_freq, feat_dir, prop_dir)

    
def reduce_feat(query, gallery):
    #query = F.normalize(query)
    #gallery = F.normalize(gallery)
    #'''
    _, raw_feat_dim = query.shape
    if raw_feat_dim < 512:
        reduce_feat_dim = raw_feat_dim
    else:
        reduce_feat_dim = 512
    pca = sklearn.decomposition.PCA(reduce_feat_dim)
    pca.fit(gallery)
    query = pca.transform(query)
    gallery = pca.transform(gallery)
    query = torch.tensor(query)
    gallery = torch.tensor(gallery)
    query = F.normalize(query)
    gallery = F.normalize(gallery)
    #'''
    return query, gallery

def gen_result_file(result_file, query_names, gallery_names, nn_idx):
    with open(os.path.join(result_file), 'w') as f:
        for i in range(len(query_names)):
            query_name = query_names[i]
            idx = nn_idx[i]
            nn_name = list(gallery_names[idx])
            nn_name = [x + '.jpg' for x in nn_name]
            f.write(query_name + '.jpg')
            f.write(',{')
            f.write(','.join(nn_name))
            f.write('}\n')


def load_feat(feat_dir):
    name_list = os.listdir(feat_dir)
    feat = [torch.load(os.path.join(feat_dir, x)) for x in name_list]
    feat = torch.stack(feat)
    return name_list, feat

def nn_search(query_vectors, gallery_vectors, print_freq):
    nn_idx = []
    query_dataset = torch.utils.data.TensorDataset(query_vectors)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    if gallery_vectors is None:
        gallery_vectors = query_vectors.cuda()
        retrieval_query = True
    else:
        gallery_vectors = gallery_vectors.cuda()
        retrieval_query = False

    def get_idx(query, gallery_vectors, retrieval_query):
        dist_matrix = torch.cdist(query, gallery_vectors)
        # cosine
        #dist_matrix = -torch.matmul(query, gallery_vectors.t())
        if retrieval_query:
            idx = dist_matrix.topk(11, largest=False)[1]
            idx = idx[:, 1:]
        else:
            idx = dist_matrix.topk(10, largest=False)[1]
        return idx

    for i, data in enumerate(query_loader):
        query = data[0]
        query = query.cuda()
        idx = get_idx(query, gallery_vectors, retrieval_query)
        nn_idx.append(idx)
        if i % print_freq == 0:
            print("NN Search: {}/{}".format(i, len(query_loader)))
    nn_idx = torch.cat(nn_idx, dim=0)
    return nn_idx

def save_feature(loader, model, epoch, LOG, print_freq, feat_dir, prop_dir):
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(prop_dir, exist_ok=True)
    start_time = time.time()
    meters = AverageMeterSet()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data in enumerate(loader):
        if len(data) == 2:
            input, name = data
        elif len(data) == 3:
            input, target, name = data

        meters.update('data_time', time.time() - end)

        if args.tencrops:
            bs, ncrops, c, h, w = input.size()
            input = input.view(-1, c, h, w)

        with torch.no_grad():
            # compute output
            model_out = model(input)
            if isinstance(model_out, tuple):
                backbone_logit, cluster_logit, class_logit, feat = model_out
            else:
                class_logit = model_out

        if args.tencrops:
            class_logit = class_logit.view(bs, ncrops, -1).mean(1)
            prop = F.softmax(class_logit, dim=1)
            feat = feat.view(bs, ncrops, -1).mean(1)
            feat = F.normalize(feat)
        else:
            prop = F.softmax(class_logit, dim=1)
            feat = F.normalize(feat)

        for j in range(len(name)):
            torch.save(feat[j].cpu(), os.path.join(feat_dir, name[j]))
            torch.save(prop[j].cpu(), os.path.join(prop_dir, name[j]))

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            LOG.info(
                'Extract Feat: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                .format(i, len(loader), meters=meters))

    LOG.info("--- save feature in {} seconds ---".format((time.time() - start_time)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
