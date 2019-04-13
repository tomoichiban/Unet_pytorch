from datetime import datetime
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


from model import *
from util import *
from VOCdataset import *

data_path = './'
save_path = './Unet_param'
result_path = './image/result.jpg'
batch_size = 8

device = torch.device('cuda')
print('Device is ' + str(device))


transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
train_set = VOCSegDataset(root =data_path, train=True, crop_size=(256, 256), transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False)
valid_set = VOCSegDataset(root =data_path, train=False, crop_size=(256, 256), transform=transform)
valid_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False)

net = Unet().to(device)

if os.path.exists(save_path):
    net = torch.load(save_path)
else:
    net = Unet().to(device)

criterion = nn.MSELoss().to(device)

optimizer = torch.optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0.01, weight_decay=0.001)

if os.path.exists('loss_y'):
    loss_y = torch.load('loss_y')
else:
    loss_y = []


for epoch in range(5):
    for img, label, _ in train_loader:
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        # forward
        output = net(img)
        loss = criterion(output, label)

        # backward
        loss.backward()
        optimizer.step()

    loss_y.append(loss.item())
    print('epoch ', str(epoch + 1), 'finish. Loss:', loss.item())

torch.save(net, save_path)
torch.save(loss_y, 'loss_y')


plt.plot(loss_y)
if not os.path.exists('./image'):
    os.mkdir('./image')
plt.savefig('image/loss.jpg', format='jpg')
plt.close()

plt.subplot(3,1,1)
img, label, _ = iter(valid_loader).next()
output_valid = net(img.cuda())
imshow(torchvision.utils.make_grid(img, nrow = 4))

plt.subplot(3,1,2)
imshow(torchvision.utils.make_grid(_, nrow = 4))

plt.subplot(3,1,3)
output_valid = output_valid.detach().cpu()
output1 = torch.zeros(batch_size,1,256,256)
output1[output_valid[:,0:1,:,:] > output_valid[:,1:2,:,:]] = 1
imshow(torchvision.utils.make_grid(output1, nrow = 4))

plt.savefig(result_path, format='jpg')



