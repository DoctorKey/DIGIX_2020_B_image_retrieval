import sys
import torch
import os
import shutil

def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el)
                                                            for el in lst)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1.):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1.):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class Vis_line:
    def __init__(self, vis, title):
        self.vis = vis
        self.title = title
        self.win = None

    def update(self, x, y):
        if self.win:
            self.vis.line(X=torch.tensor([x]), Y=torch.tensor([y]), win=self.win, update="append")
        else:
            self.win = self.vis.line(X=torch.tensor([x]), Y=torch.tensor([y]), opts=dict(title=self.title))


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def parameter_count(module):
    return sum(int(param.numel()) for param in module.parameters())


def save_checkpoint(state, is_best, dirpath, epoch, LOG, fp16=False):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to {} ---".format(checkpoint_path))
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to {} ---".format(best_path))
    if fp16:
        sd = state['state_dict']
        arch = state['arch']
        l = ['module.backbone.fc.weight', 'module.backbone.fc.bias', 
            'module.backbone.feat_fc.weight', 'module.backbone.feat_fc.bias', 
            'module.metric_fc.weight']
        for k in l:
            sd.pop(k, None)
        for k in sd.keys():
            sd[k] = sd[k].half()
        state['state_dict'] = sd
        torch.save(state, os.path.join(dirpath, 'checkpoint.{}.fp16.ckpt'.format(epoch)))
        

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        labeled_minibatch_size = max(target.ne(-1).sum(), 1e-8).item()

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            #res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
            res.append(correct_k.mul_(100.0 / labeled_minibatch_size)[0])
    return res

def Prec1(outputs, targets):
    """Computes the precision@1"""
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    num = targets.size(0)
    prec1 = correct * 100.0 / num
    return prec1
    
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()
        #m.weight.requires_grad_(False)
        #m.bias.requires_grad_(False)