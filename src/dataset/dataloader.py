import math
import random
import torch
import torchvision

import os
import json

from PIL import Image


import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def create_ImbalancedSampler_loader(train_dataset, args):
    def get_label(dataset, idx):
        return dataset.imgs[idx][1]
    train_sample = ImbalancedDatasetSampler(train_dataset, callback_get_label=get_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        sampler=train_sample, 
        batch_size=args.batch_size, 
        num_workers=args.workers)
    return train_loader

def create_train_loader(train_dataset, args):
    train_loader = torch.utils.data.DataLoader(train_dataset,
             batch_size=args.batch_size,
             shuffle=True,
             num_workers=args.workers,
             pin_memory=True,
             drop_last=True)
    return train_loader

def create_eval_loader(eval_dataset, args):
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    return eval_loader


def create_SameRatioSampler_loader(train_dataset, args):
    train_sample = SameRatioSampler(train_dataset, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_sample,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               batch_size=1,
                                               pin_memory=True,
                                               drop_last=False)
    return train_loader

def create_SameRatioSampler_eval_loader(eval_dataset, args):
    print("Attention! eval at batch=1")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,#args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    return eval_loader


class MPerClassSampler(Sampler):
    def __init__(self, labels, batch_size, m=4):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size
        self.m = m
        assert batch_size % m == 0, 'batch size must be divided by m'

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.__len__()):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int)

            while inds.shape[0] < self.batch_size:
                sample_label = np.random.choice(self.labels_unique)
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                sample_label_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
                subsample = np.random.permutation(sample_label_ids)[:self.m]
                inds = np.append(inds, subsample)

            inds = inds[:self.batch_size]
            inds = np.random.permutation(inds)
            yield list(inds)
            

class KPerClassSampler(Sampler):
    def __init__(self, labels, batch_size, k=4, drop_last=False):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.c2id = self.get_class_to_idlist()
        self.batch_size = batch_size
        self.k = k
        self.drop_last = drop_last
        assert batch_size % k == 0, 'batch size must be divided by k'

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        else:
            return math.ceil(len(self.labels) / self.batch_size)


    def __iter__(self):
        sample_c2id = self.get_class_to_idlist()
        while len(sample_c2id) > 0:
            labels_in_batch = set()
            inds = []

            while len(inds) < self.batch_size:
                remain_class = set(sample_c2id.keys()) - labels_in_batch
                if len(remain_class) > 0:
                    sample_label = random.choice(list(remain_class))
                    labels_in_batch.add(sample_label)
                    id_list = sample_c2id[sample_label]
                    if len(id_list) > self.k:
                        inds.extend(id_list[:self.k])
                        sample_c2id[sample_label] = id_list[self.k:]
                    else:
                        while len(id_list) != self.k:
                            if len(self.c2id[sample_label]) <= self.k:
                                idx = random.choice(self.c2id[sample_label])
                            else:
                                remain_idx = list(set(self.c2id[sample_label]) - set(id_list))
                                idx = random.choice(remain_idx)
                            id_list.append(idx)
                        inds.extend(id_list)
                        sample_c2id.pop(sample_label)
                else:
                    sample_label = random.choice(self.labels_unique)
                    if sample_label in labels_in_batch:
                        continue
                    labels_in_batch.add(sample_label)
                    id_list = self.c2id[sample_label]

                    if self.k > len(id_list):
                        # with replacement
                        idx = random.choices(id_list, k=self.k)
                    else:
                        # without replacement
                        idx = random.sample(id_list, k=self.k)
                    inds.extend(idx)

                

            assert len(inds) == self.batch_size
            random.shuffle(inds)
            yield inds

    def get_class_to_idlist(self):
        d = dict()
        for idx in range(len(self.labels)):
            target = self.labels[idx]
            if d.get(target, None) is None:
                d[target] = [idx]
            else:
                d[target].append(idx)
        for t in d:
            random.shuffle(d[t])
        return d


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def show(sampler, labels):
    for i in sampler:
        r = [labels[x] for x in i]
        print('{}: {}'.format(i, r))


class SmallImageNetFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        
        img = self._pil_loader(path)
        
        name = os.path.basename(path)
        name = name.split('.')[0]
        return img, target, name
    
    def _pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
        return img  # .convert('RGB')


class SameRatioSampler(Sampler):
    
    def __init__(self, dataset, batch_size=64, drop_last=False):
        
        self.root_dir = dataset.root
        self.labels = np.array(dataset.targets)
        self.batch_size = batch_size
        self.drop_last = drop_last
        #with open(target_dir + os.sep + 'ratio.js', 'r') as f:
        #    self.ratio_data = json.loads(f.read())
        self.ratio_data = self._get_ratio_stat()
        self.ratio_key = list(self.ratio_data.keys())
        
        total = sum([len(v) for v in self.ratio_data.values()])
        self.weight = [len(v) / total for v in self.ratio_data.values()]

    def _get_ratio_stat(self):
        from .my_transforms import ratio_map
        train_dataset = SmallImageNetFolder(self.root_dir)
        ratio_dict = {
            '9/16': [],
            '3/4': [],
            '1/1': [],
            '4/3': [],
            '16/9': []
        }
        for i, it in enumerate(train_dataset):
    
            h = it[0].size[1]
            w = it[0].size[0]
    
            if h > w:
                h = int(h / w * 384)
                w = 384
            else:
                w = int(w / h * 384)
                h = 384
            ratio = h / w
    
            ratio_str_two = ratio_map(ratio)
            ratio_dict[ratio_str_two].append(i)
        print("Calculating the image height/width ratio done.")
        return ratio_dict
    
    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        else:
            return math.ceil(len(self.labels) / self.batch_size)
    
    def __iter__(self):
        for i in range(self.__len__()):
            current_ratio = np.random.choice(self.ratio_key, size=1, p=self.weight)[0]
            _temp = np.random.choice(np.array(self.ratio_data[current_ratio].copy()),
                                   size=self.batch_size, replace=False)
            
            indices = list(np.sort(_temp))
        
            yield indices
    

if __name__ == '__main__':
    labels = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
    s = MPerClassSampler(labels, 2, 2)
    ks = KPerClassSampler(labels, 4, 2)
    import IPython
    IPython.embed()