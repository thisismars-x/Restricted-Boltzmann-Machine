import torchvision
from torchvision import utils, datasets, transforms

from model import *
from helper import *

import torch

#change here if need be
VISIBLE = 784
HIDDEN = 128
GIBBS_SAMPLER = 1

model = rboltzmann(VISIBLE, HIDDEN, GIBBS_SAMPLER)

loader = torchvision.datasets.MNIST(
    root = './output',
    train = True,
    transform = transforms.Compose([transforms.ToTensor()])
)

dloader = torch.utils.data.DataLoader(loader, batch_size = 64, shuffle = True)
model = train(model, loader, 10)

'''
store 20 real and fake pics
'''

iter_loader = iter(dloader)

import os

os.mkdir('./output/real')
os.mkdir('./output/fake')

for i in range(20):
    img = next(iter_loader)[0]
    v, gibb = model(img.view(-1, 784))
    real = './output/real/' + str(i) + '.png'
    fake = './output/fake/' + str(i) + '.png'

    save(utils.make_grid(v.view(64, 1, 28, 28).data), fname = real)
    save(utils.make_grid(gibb.view(64, 1, 28, 28).data), fname = fake)





