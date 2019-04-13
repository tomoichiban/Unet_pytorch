import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def doubleCrop(img, label, size):
    i, j, h, w = transforms.RandomCrop.get_params(img, size)
    img = transforms.functional.crop(img, i, j, h, w)
    label = transforms.functional.crop(label, i, j, h, w)
    return img, label


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
