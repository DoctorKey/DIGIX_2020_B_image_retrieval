import os
import socket
import torch
import torchvision
import torchvision.transforms as transforms

from ..utils import export
from .DIGIX import ImageFold, ImageNetFolder
from .my_transforms import MyRotation, MyColorJitter


traindir = '/amax/opt/Dataset/HuaWei_DIGIX/train_data'
test_A_dir = '/amax/opt/Dataset/HuaWei_DIGIX/test_data_A'
test_B_dir = '/amax/opt/Dataset/HuaWei_DIGIX/test_data_B'

"""
@wanghy now, i mainly use this function, try to modify this part to test aug methods.
RandomErasing https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
"""

def get_normal_transforms(channel_stats, resize_size, input_size, tencrops):
    train_transformation = transforms.Compose([
        transforms.Resize(size=resize_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=input_size),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    if tencrops:
        eval_transformation = transforms.Compose([
            transforms.Resize(size=resize_size),
            transforms.TenCrop(size=input_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(**channel_stats)(crop) for crop in crops]))
        ])
    else:
        eval_transformation = transforms.Compose([
            transforms.Resize(size=resize_size),
            transforms.CenterCrop(size=input_size),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
    return train_transformation, eval_transformation

def get_new_transforms(channel_stats, resize_size, input_size, tencrops):
    train_transformation = transforms.Compose([
        transforms.Resize(size=resize_size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(30),
        transforms.RandomPerspective(p=0.5, distortion_scale=0.3, fill=(124, 117, 104)),
        MyColorJitter(0.5),
        transforms.RandomCrop(size=input_size),
        MyRotation(0.3),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])
    if tencrops:
        eval_transformation = transforms.Compose([
            transforms.Resize(size=resize_size),
            transforms.TenCrop(size=input_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(**channel_stats)(crop) for crop in crops]))
        ])
    else:
        eval_transformation = transforms.Compose([
            transforms.Resize(size=resize_size),
            transforms.CenterCrop(size=input_size),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
    return train_transformation, eval_transformation


@export
def DIGIX(args=None):
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    

    resize_size = args.resize_size
    input_size = args.input_size

    #train_transformation, eval_transformation = get_normal_transforms(channel_stats, resize_size, input_size, args.tencrops)
    train_transformation, eval_transformation = get_new_transforms(channel_stats, resize_size, input_size, args.tencrops)

    train_dataset = ImageNetFolder(traindir, train_transformation)
    query = ImageFold(os.path.join(test_A_dir, 'query'), eval_transformation)
    gallery = ImageFold(os.path.join(test_A_dir, 'gallery'), eval_transformation)
    extract_train = ImageNetFolder(traindir, eval_transformation)

    return {
        'train_dataset': train_dataset,
        'val_dataset': None,
        'query_dataset': query,
        'gallery_dataset': gallery,
        'extract_train': extract_train,
        'label': None,
        'num_classes': 3094,
    }

@export
def DIGIX_test_B(args=None):
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    resize_size = args.resize_size
    input_size = args.input_size

    #train_transformation, eval_transformation = get_normal_transforms(channel_stats, resize_size, input_size, args.tencrops)
    train_transformation, eval_transformation = get_new_transforms(channel_stats, resize_size, input_size, args.tencrops)

    train_dataset = ImageNetFolder(traindir, train_transformation)
    query = ImageFold(os.path.join(test_B_dir, 'query'), eval_transformation)
    gallery = ImageFold(os.path.join(test_B_dir, 'gallery'), eval_transformation)
    extract_train = ImageNetFolder(traindir, eval_transformation)

    return {
        'train_dataset': train_dataset,
        'val_dataset': None,
        'query_dataset': query,
        'gallery_dataset': gallery,
        'extract_train': None,
        'label': None,
        'num_classes': 3094,
    }




def get_dataset_config(dataset_name, args=None):
    dataset_factory = globals()[dataset_name]
    params = dict(args=args)
    dataset_config = dataset_factory(**params)
    return dataset_config
    
