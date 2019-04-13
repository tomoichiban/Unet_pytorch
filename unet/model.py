import torch
import torch.nn as nn
import torchvision
from util import *

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.block(input)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.contract1 = UnetBlock(3, 64)
        self.contract2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),UnetBlock(64, 128))
        self.contract3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),UnetBlock(128, 256))
        self.contract4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),UnetBlock(256, 512))
        self.contract5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            UnetBlock(512, 1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        )
        self.expand1 = nn.Sequential(
            UnetBlock(1024, 512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        )
        self.expand2 = nn.Sequential(
            UnetBlock(512, 256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        )
        self.expand3 = nn.Sequential(
            UnetBlock(256, 128),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        )
        self.expand4 = UnetBlock(128, 64)

        self.last = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        c1 = self.contract1(input)
        c2 = self.contract2(c1)
        c3 = self.contract3(c2)
        c4 = self.contract4(c3)
        mid = self.contract5(c4)
        e1 = self.expand1(torch.cat((c4, mid), dim=1))
        e2 = self.expand2(torch.cat((c3, e1), dim=1))
        e3 = self.expand3(torch.cat((c2, e2), dim=1))
        e4 = self.expand4(torch.cat((c1, e3), dim=1))
        output = self.last(e4)

        return output


