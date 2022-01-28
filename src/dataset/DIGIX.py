import os
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageFold(torch.utils.data.Dataset):
    """docstring for ImageFold"""
    def __init__(self, root, transform=None):
        super(ImageFold, self).__init__()
        self.root = os.path.expanduser(root)
        self._transform = transform
        self._image_filename = os.listdir(self.root)

    def __getitem__(self, index):
        image_filename = self._image_filename[index]
        path = os.path.join(self.root, image_filename)
        sample = pil_loader(path)
        if self._transform is not None:
            sample = self._transform(sample)
        return sample, image_filename.split('.')[0]

    def __len__(self):
        return len(self._image_filename)

class ImageNetFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        name = os.path.basename(path)
        name = name.split('.')[0]
        return img, target, name



if __name__ == "__main__":
    import IPython
    IPython.embed()
