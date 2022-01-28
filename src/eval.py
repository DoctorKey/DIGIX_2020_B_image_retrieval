import re
import os
import shutil
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import AverageMeterSet, accuracy


def validate_cgd_digix(eval_loader, model, epoch, writer=None, LOG=None, print_freq=100):
    start_time = time.time()

    class_criterion = nn.CrossEntropyLoss().cuda()
    #class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    query_vectors = []
    query_labels = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data in enumerate(eval_loader):
        if len(data) == 2:
            input, target = data
        elif len(data) == 3:
            input, target, _ = data

        meters.update('data_time', time.time() - end)

        with torch.no_grad():
            target = target.cuda()

            # compute output
            model_out = model(input)
            if isinstance(model_out, tuple):
                backbone_logit, cluster_logit, class_logit, feature = model_out
            else:
                class_logit = model_out

            query_vectors.append(feature.cpu())
            query_labels.append(target.cpu())


        minibatch_size = len(target)
        class_loss = class_criterion(class_logit, target)
        meters.update('class_loss', class_loss.item(), minibatch_size)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(class_logit.data, target.data, topk=(1, 5))
        meters.update('top1', prec1, minibatch_size)
        meters.update('top5', prec5, minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    i, len(eval_loader), meters=meters))

    
    query_vectors = torch.cat(query_vectors, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    s, top1, top10 = retrieval_score_withlabel(query_vectors, query_labels, None, None, print_freq)

    LOG.info(' * TEST Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))
    LOG.info(' * Retrieval Score:{:.4f} Top1:{:.4f} Top10:{:.4f}'.format(s, top1, top10))

    if writer:
        writer.add_scalar("val/prec1", meters['top1'].avg, epoch)
        writer.add_scalar("val/prec5", meters['top5'].avg, epoch)

    LOG.info("--- validation in {} seconds ---".format((time.time() - start_time)))
    return meters['top1'].avg

def retrieval_score_withlabel(query_vectors, query_labels, gallery_vectors, gallery_labels, print_freq):
    top1_correct_num = 0
    top10_correct_num = 0
    query_dataset = torch.utils.data.TensorDataset(query_vectors, query_labels)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=128,
                                        shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    if gallery_vectors is None:
        gallery_vectors = query_vectors.cuda()
        gallery_labels = query_labels
        retrieval_query = True
    else:
        gallery_vectors = gallery_vectors.cuda()
        retrieval_query = False

    def get_idx(query, gallery_vectors, retrieval_query):
        dist_matrix = torch.cdist(query, gallery_vectors)
        if retrieval_query:
            idx = dist_matrix.topk(11, largest=False)[1]
            idx = idx[:, 1:]
        else:
            idx = dist_matrix.topk(10, largest=False)[1]
        return idx

    num_query = len(query_labels)
    for i, data in enumerate(query_loader):
        query, query_label = data
        query = query.cuda()
        idx = get_idx(query, gallery_vectors, retrieval_query)
        nn_label = gallery_labels[idx]
        top1_correct_num += (nn_label[:, 0] == query_label).sum().item()
        top10_correct_num += (nn_label == query_label.reshape(-1, 1)).sum().item()
        if i % print_freq == 0:
            print("NN Search: {}/{}".format(i, len(query_loader)))
    top1 = float(top1_correct_num) / num_query
    top10 = float(top10_correct_num) / (num_query * 10)
    s = 0.5 * top1 + 0.5 * top10
    return s, top1, top10

def eval_retrieval_by_labelfile(result_file, label_file, LOG):
    with open(label_file) as f:
        label = f.readlines()

    image_to_class = dict()
    for l in label:
        l = l.rstrip()
        file, target = l.split(',')
        file = file.split('/')[-1]
        image_to_class[file] = target

    with open(result_file) as f:
        result = f.readlines()

    top1_count = 0
    top10_count = 0
    for r in result:
        r = r.rstrip()
        q, nn = r.split('{')
        nn = nn.split('}')[0]
        nn_list = nn.split(',')
        q = q.split(',')[0]

        target = image_to_class[q]
        if image_to_class[nn_list[0]] == target:
            top1_count += 1
        for n in nn_list:
            if image_to_class[n] == target:
                top10_count += 1

    top1 = float(top1_count) / len(result)
    top10 = float(top10_count) / (len(result) * 10)
    score = 0.5 * top1 + 0.5 * top10
    LOG.info(' * Score:{:.4f} Top1:{:.4f} Top10:{:.4f}'.format(score, top1, top10))
    return score
