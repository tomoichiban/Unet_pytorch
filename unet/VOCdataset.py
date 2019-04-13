import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

from util import *

class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''

    def __init__(self, root, train, crop_size=(256,256), transform=None):
        self.crop_size = crop_size
        self.transform = transform
        data_list, label_list = self.__read_images(root, train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')

    def _filter(self, images):  # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    # 加载图片
    def __getitem__(self, index):
        img = self.data_list[index]
        label = self.label_list[index]

        img = Image.open(img)
        label = Image.open(label)
        img, label = doubleCrop(img, label, self.crop_size)
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        label[label > 0] = 1
        label_true = torch.zeros(label.shape)
        label_true[label > 0] = 1
        label_false = torch.zeros(label.shape)
        label_false[label == 0] = 1
        label_unet = torch.cat((label_true, label_false), dim=0)

        return img, label_unet, label

    def __len__(self):
        return len(self.data_list)

    # 读取图片列表
    def __read_images(self, root, train=True):
        root = root + '/VOCdevkit/VOC2012'
        root = os.path.expanduser(root)
        txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
        with open(txt_fname, 'r') as f:
            images = f.read().split()
        data = []
        for i in images:
            data.append(os.path.join(root, 'JPEGImages', i + '.jpg'))
        label = []
        for i in images:
            label.append(os.path.join(root, 'SegmentationClass', i + '.png'))
        return data, label