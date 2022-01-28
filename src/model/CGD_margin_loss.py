import torch
from torch import nn
from torch.nn import functional as F
from ..losses import ArcMarginProduct, SphereProduct, L2FC, MarginCosineProduct, FC

class GlobalDescriptor(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, x):
        assert x.dim() == 4, 'the input tensor of GlobalDescriptor must be the shape of [B, C, H, W]'
        if self.p == 1:
            return x.mean(dim=[-1, -2])
        elif self.p == float('inf'):
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            sum_value = x.pow(self.p).mean(dim=[-1, -2])
            return torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / self.p))

    def extra_repr(self):
        return 'p={}'.format(self.p)


class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim() == 2, 'the input tensor of L2Norm must be the shape of [B, C]'
        return F.normalize(x, p=2, dim=-1)

class CGD_margin_loss(nn.Module):
    def __init__(self, backbone, gd_config, feature_dim, num_classes, args):
        super().__init__()
        self.backbone = backbone
        self.backbone_feat_dim = backbone.feat_map_channel

        # Main Module
        n = len(gd_config)
        k = feature_dim // n
        assert feature_dim % n == 0, 'the feature dim should be divided by number of global descriptors'

        self.normalize = True
        if args.metric == 'arc_margin':
            metric_fc = ArcMarginProduct(feature_dim, num_classes, s=args.feat_norm, m=args.margin, easy_margin=args.easy_margin)
        elif args.metric == 'cos_margin':
            metric_fc = MarginCosineProduct(feature_dim, num_classes, s=args.feat_norm, m=args.margin)
        elif args.metric == 'sphere':
            metric_fc = SphereProduct(feature_dim, num_classes, m=args.margin)
        elif args.metric == 'l2fc':
            metric_fc = L2FC(feature_dim, num_classes)
        else:
            metric_fc = FC(feature_dim, num_classes)
            self.normalize = False
        self.metric_fc = metric_fc

        self.global_descriptors = []
        for i in range(n):
            if gd_config[i] == 'S':
                p = 1
            elif gd_config[i] == 'M':
                p = float('inf')
            else:
                p = 3
            if self.normalize:
                layer = nn.Sequential(GlobalDescriptor(p=p), 
                    nn.Linear(self.backbone_feat_dim[-1], k, bias=True), 
                    nn.BatchNorm1d(k), 
                    L2Norm())
            else:
                layer = nn.Sequential(GlobalDescriptor(p=p), 
                    nn.Linear(self.backbone_feat_dim[-1], k, bias=True), 
                    nn.BatchNorm1d(k))
            self.global_descriptors.append(layer)


        self.global_descriptors = nn.ModuleList(self.global_descriptors)

 
    def forward(self, x, target=None):
        backbone_logit, backbone_f, feat_maps = self.backbone(x)
        global_descriptors = []
        feat_map = feat_maps[-1]
        for i in range(len(self.global_descriptors)):
            global_descriptor = self.global_descriptors[i](feat_map)
            global_descriptors.append(global_descriptor)
        global_descriptors = torch.cat(global_descriptors, dim=-1)
        if self.normalize:
            global_descriptors = F.normalize(global_descriptors)
        logit = self.metric_fc(global_descriptors, target)
        return backbone_logit, backbone_f, logit, global_descriptors