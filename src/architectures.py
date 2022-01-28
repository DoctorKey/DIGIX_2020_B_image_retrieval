import os
import torch
import torch.nn as nn
import torchvision

from .model.resnet import Bottleneck
from .model.resnet import ResNet

from .model.fishnet import fish

from .model.dla import DLA, BottleneckX

from .model.hrnet import cls_net

from .utils import export, parameter_count


@export
def resnet101(pretrained=False, num_classes=200, pretext_classes=4, **kwargs):
    assert not pretrained
    model = ResNet(Bottleneck, [3,4,23,3], num_classes, pretext_classes, **kwargs)
    return model

@export
def fishnet99(pretrained=False, num_classes=1000, pretext_classes=4, **kwargs):
    """
    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |     |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 2, 6, 2, 1, 1, 1, 1, 2, 2],
        'num_trans_blks': [1, 1, 1, 1, 1, 4],
        'num_cls': num_classes,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)

@export
def dla102x(pretrained=False, num_classes=1000, pretext_classes=4, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                num_classes=num_classes, pretext_classes=pretext_classes, block=BottleneckX, residual_root=True, **kwargs)
    return model


@export
def hrnet_w18(pretrained=False, num_classes=1000, pretext_classes=4, **kwargs):
    net_cfg = {
        "num_cls": num_classes,
        'cfg': './src/model/cfg/cls_hrnet_w18.yaml',
    }
    return cls_net(**net_cfg)


@export
def hrnet_w30(pretrained=False, num_classes=1000, pretext_classes=4, **kwargs):
    net_cfg = {
        "num_cls": num_classes,
        'cfg': './src/model/cfg/cls_hrnet_w30.yaml',
    }
    return cls_net(**net_cfg)


def create_model(model_name, num_classes, pretext_classes=4, detach_para=False, DataParallel=True, **kwargs):
    model_factory = globals()[model_name]
    model_params = dict(pretrained=False, num_classes=num_classes, pretext_classes=pretext_classes, **kwargs)
    model = model_factory(**model_params)
    if DataParallel:
        model = nn.DataParallel(model.cuda())
    else:
        model = model.cuda()

    if detach_para:
        for param in model.parameters():
            param.detach_()
    return model


def load_pretrained(model, pretrained, arch, LOG, DataParallel=True):
    if os.path.isfile(pretrained):
        LOG.info("=> loading pretrained from checkpoint {}".format(pretrained))
        if DataParallel:
            model = nn.DataParallel(model)
        checkpoint = torch.load(pretrained)
        state_dict = checkpoint['state_dict']
        ret = model.load_state_dict(state_dict, strict=False)
        LOG.info("=> loaded pretrained {}".format(pretrained))
        LOG.info("=> not load keys {} ".format(ret))
    elif pretrained.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(pretrained)
        if 'vgg' in arch:
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        elif 'fish' in arch:
            state_dict = state_dict['state_dict']
            state_dict.pop('module.fish.fish.9.4.1.weight', None)
            state_dict.pop('module.fish.fish.9.4.1.bias', None)
            state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
        elif 'hrnet' in arch:
            state_dict.pop('classifier.weight')
            state_dict.pop('classifier.bias')
        else:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        if DataParallel:
            model = nn.DataParallel(model)
        LOG.info("=> loaded pretrained {} ".format(pretrained))
        LOG.info("=> not load keys {} ".format(ret))
    else:
        if DataParallel:
            model = nn.DataParallel(model)
        LOG.info("=> NOT load pretrained")
    return model