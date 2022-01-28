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

from .. import losses, ramps, cli
from ..eval import validate_cgd_digix
from ..model.CGD_margin_loss import CGD_margin_loss
from ..architectures import create_model, load_pretrained
from ..run_context import RunContext
from ..dataset.datasets import get_dataset_config
from ..dataset.dataloader import create_train_loader, create_eval_loader
from ..utils import save_checkpoint, AverageMeterSet, parameters_string, Prec1, set_bn_eval

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
    train_dataset = dataset_config.get('train_dataset')
    val_dataset = dataset_config.get('val_dataset', None)
    num_classes = dataset_config.get('num_classes')
    train_loader = create_train_loader(train_dataset, args=args)
    if val_dataset is not None:
        eval_loader = create_eval_loader(val_dataset, args=args)
    else:
        eval_loader = None

    LOG.info("=> creating model '{arch}'".format(arch=args.arch))
    backbone = create_model(args.arch, num_classes, args.pretext_class, detach_para=False, DataParallel=False)
    backbone = load_pretrained(backbone, args.pretrained, args.arch, LOG, DataParallel=False)
    cgd = CGD_margin_loss(backbone, args.gd_config, args.feature_dim, num_classes, args)
    LOG.info(parameters_string(cgd))
    cgd = nn.DataParallel(cgd).cuda()


    optimizer = torch.optim.Adam(cgd.parameters(), lr=args.lr)

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        train(train_loader, cgd, optimizer, epoch, context.vis_log)

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0 and eval_loader is not None:
            prec1 = validate_cgd_digix(eval_loader, cgd, epoch, context.vis_log, LOG, args.print_freq)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': cgd.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1, LOG)

    save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': cgd.state_dict(),
                'best_prec1': best_prec1,
            }, False, checkpoint_path, 'final', LOG, fp16=True)
    LOG.info("best_prec1 {}".format(best_prec1))


def train(train_loader, model, optimizer, epoch, writer):
    global global_step
    start_time = time.time()

    class_criterion = nn.CrossEntropyLoss()

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    #model.apply(set_bn_eval)

    end = time.time()
    for i, data in enumerate(train_loader):
        if len(data) == 3:
            input, target, _ = data
        elif len(data) == 2:
            input, target = data
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))   

        input, target = input.cuda(), target.cuda()
        
        model_out = model(input, target)
        if isinstance(model_out, tuple):
            backbone_logit, cluster_logit, class_logit, feat = model_out
        else:
            class_logit = model_out

        class_loss = class_criterion(class_logit, target)
        loss = class_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        meters.update('lr', optimizer.param_groups[0]['lr'])
        minibatch_size = len(target)
        meters.update('loss', loss.item(), minibatch_size)
        meters.update('class_loss', class_loss.item(), minibatch_size)
        prec1 = Prec1(class_logit.data, target.data)
        meters.update('top1', prec1, minibatch_size)   

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Loss {meters[loss]:.4f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'.format(
                    epoch, i, len(train_loader), meters=meters))

    LOG.info(' * TRAIN Prec@1 {top1.avg:.3f} ({0:.1f}/{1:.1f})'
          .format(meters['top1'].sum / 100, meters['top1'].count, top1=meters['top1']))
    if writer is not None:
        writer.add_scalar("train/lr", meters['lr'].avg, epoch)
        writer.add_scalar("train/class_loss", meters['class_loss'].avg, epoch)
        writer.add_scalar("train/prec1", meters['top1'].avg, epoch)

    LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    if args.lr_rampup != 0:
        lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    if args.lr_reduce_epochs:
        reduce_epochs = [int(x) for x in args.lr_reduce_epochs.split(',')]
        for ep in reduce_epochs:
            if epoch >= ep:
                lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
