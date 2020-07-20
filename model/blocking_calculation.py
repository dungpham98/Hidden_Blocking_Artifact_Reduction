import os
from os import listdir
from os.path import isfile, join
import time
import torch
import csv
import numpy as np
import utils
import logging
from collections import defaultdict
from torchvision.utils import save_image
from options import *
from model.hidden import Hidden
from average_meter import AverageMeter

def blocking_value(encoded_imgs,batch,block_size,block_number):
    #blocking effect value
    Total = 0
    Vcount = 0
    Hcount = 0
    V_average = 0
    H_average = 0
    for idx in range(0,batch):
        V_average = 0
        H_average = 0
        for i in range(0,len(encoded_imgs)-1):
            if((i+1) % block_number != 0):
                img = encoded_imgs[i][idx][0].cpu().detach().numpy()
                img_next = encoded_imgs[i+1][idx][0].cpu().detach().numpy()
                for j in range(0,block_size):
                    distinct = np.abs(img[j][block_size-1]-img_next[j][0])
                    V_average = V_average+distinct
                    Total = Total +1
                    if(distinct > 0.25):
                        Vcount = Vcount+1

        
        for i in range(0,len(encoded_imgs)-block_number):
            img = encoded_imgs[i][idx][0].cpu().detach().numpy()
            img_next = encoded_imgs[i+block_number][idx][0].cpu().detach().numpy()
            for j in range(0,block_size):
                distinct = np.abs(img[block_size-1][j]-img_next[0][j])
                H_average = H_average+distinct
                Total = Total + 1
                if(distinct > 0.25):
                    Hcount = Hcount+1
    
    blocking_loss = (Vcount+Hcount)/(Total)
    return blocking_loss

validation_folder = ""
batch_size = 16
block_size = 32
H = 128
W = 128

data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomCrop((H, W), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop((H, W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

validation_images = datasets.ImageFolder(validation_folder, data_transforms['test'])
validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=batch_size, drop_last=True,
                                                shuffle=False, num_workers=4)


for image,_ in validation_loader:
    imgs,_ = utlis.concatImgs(imgs,block_number)
    blocking_val = blocking_value(imgs,batch)
