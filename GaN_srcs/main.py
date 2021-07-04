import os
import glob
import time
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from model import MainModel
from functions import train_model, test_model
from data_loader import make_dataloaders
from generator_res_unet import build_res_unet, pretrain_generator


path = "/flower_500"
test_path = "/flower_test_10"


paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
test_paths = glob.glob(test_path + "/*.jpg") # Grabbing all the image file names
np.random.seed(123)
paths_subset = np.random.choice(paths, 500, replace=False) # choosing 1000 images randomly
rand_idxs = np.random.permutation(500)
train_idxs = rand_idxs[:450] # choosing the first 8000 as training set
val_idxs = rand_idxs[450:] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')
np.random.seed(123)
test_paths_subset = np.random.choice(test_paths, 3, replace=False)
rand_idxs = np.random.permutation(3)
test_idxs = rand_idxs[:3]
TEST_paths = test_paths_subset[test_idxs]
test_dl = make_dataloaders(paths=TEST_paths, split='test')
data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
data = next(iter(val_dl))
Ls, abs_ = data['L'], data['ab']
data = next(iter(test_dl))
Ls, abs_ = data['L'], data['ab']

intend_model = "unet"

loss_D_fake = []
loss_D_real = []
loss_D = []
loss_G_GAN = []
loss_G_L1 = []
loss_G = []

if (intend_model == "unet"):
    model_x = MainModel()
    loss_D_fake, loss_D_real, loss_D, loss_G_GAN, loss_G_L1, loss_G = train_model(model_x,train_dl, val_dl, 50)

    ax = plt.subplot(2,3,1)
    ax.plot(loss_D_fake)
    bx = plt.subplot(2,3,2)
    bx.plot(loss_D_real)
    cx = plt.subplot(2,3,3)
    cx.plot(loss_D)
    dx = plt.subplot(2,3,4)
    dx.plot(loss_G_GAN)
    ex = plt.subplot(2,3,5)
    ex.plot(loss_G_L1)
    fx = plt.subplot(2,3,6)
    fx.plot(loss_G)

    test_model(model_x,test_dl)

elif (intend_model == "res_unet"):
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    pretrain_generator(net_G, train_dl, opt, criterion, 20)
    torch.save(net_G.state_dict(), "res18-unet.pt")

    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)
    loss_D_fake, loss_D_real, loss_D, loss_G_GAN, loss_G_L1, loss_G = train_model(model, train_dl, val_dl, 100)

    ax = plt.subplot(2,3,1)
    ax.plot(loss_D_fake)
    bx = plt.subplot(2,3,2)
    bx.plot(loss_D_real)
    cx = plt.subplot(2,3,3)
    cx.plot(loss_D)
    dx = plt.subplot(2,3,4)
    dx.plot(loss_G_GAN)
    ex = plt.subplot(2,3,5)
    ex.plot(loss_G_L1)
    fx = plt.subplot(2,3,6)
    fx.plot(loss_G)

    test_model(model,test_dl)

else:
    print("Error! Please enter either \"unet\" or \"res_unet\"!")