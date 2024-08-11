import random
import numpy as np
import torch
from PIL import ImageOps
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

class Datasets(Dataset):

    def __init__(self,datasets,transform=None,should_invert=True,train=False,mnist=True):
        self.datasets = datasets
        self.transform = transform
        self.should_invert = should_invert
        self.train = train
        self.mnist = mnist

    def __getitem__(self, item):
        plus = 5
        if self.mnist:
            plus += 1
        b = 10000
        if self.train :
            b *= 5

        num0 = int(random.randint(0,b))
        img0_tuple = self.datasets.__getitem__(num0)
        sameORnot = random.randint(0,1)
        if sameORnot:
            while True:
                num1 = int(random.randint(0,b))
                img1_tuple = self.datasets.__getitem__(num1)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                num1 = int(random.randint(0,1))
                img1_tuple = self.datasets.__getitem__(num1)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        channel = 'RGB'
        if self.mnist:
            channel = 'L'
        img0, img1 = img0_tuple[0].convert(channel), img1_tuple[0].convert(channel)

        if self.should_invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])],
                                                     dtype=np.float32))

    def __len__(self):
        return len(self.datasets.__getitem__(0))

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(8,8,text,style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

