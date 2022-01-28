import torchvision.transforms.functional as tF
import torchvision.transforms as transforms
import random
import torch


class MyRotation:
    """
    rotate the img with probability p,
    rotation angle contains 90, 180 and -90
    more details in https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomRotation
    """
    def __init__(self, p):
        assert 0<=p<=1
        self.p = p

    @staticmethod
    def get_params():
        rotate_list = [-90, 90, 180]
        rand = random.randint(0,2)
        return rotate_list[rand]

    def __call__(self, img):
        angle = self.get_params()

        rand = random.random()
        if torch.rand(1) > self.p:
            return img

        return tF.rotate(img, angle)

    def __repr__(self):
        return "self-defined rotation transforms"


class MyColorJitter:
    """
    A decorator for ColorJitter, we have probability p to keep the original image color
    """
    def __init__(self, p):
        assert 0<=p<=1
        self.p = p
        self.jitter = transforms.ColorJitter(hue=[-0.1, 0.1], contrast=[0.5, 1.5], brightness=[0.5, 1.5], saturation=[0.5, 2])
    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img
        else:
            return self.jitter(img)



