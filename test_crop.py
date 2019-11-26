import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.random
import torchvision.utils
def cropImg(size,img_tensor):
	imgs=[]
	batch = int(img_tensor.shape[0])
	channel = int(img_tensor.shape[1])
	h = int(img_tensor.shape[2])
	w = int(img_tensor.shape[3])
	n = int(h/size)
	print(w)
	i = 0
	while(i*size < h):
		j = 0
		while(j*size < w):
			i_n =int(i*size)
			j_n = int(j*size)
			img = img_tensor[0:batch,0:channel,i_n:(i_n+size),j_n:(j_n+size)]
			print(i_n)
			imgs.append(img)
			#torchvision.utils.save_image(img,"cropped"+str(i_n)+str(j_n)+".jpg")
			j = j + 1 
		i = i + 1
	return imgs

def concatImgs(imgs):
	img_len = len(imgs)
	i = 0
	img_cat =[]
	while(i < 16):
		img_cat.append(torch.cat([imgs[0+i],imgs[1+i],imgs[2+i],imgs[3+i]],dim=0))
		i = i + 4
	img = torch.cat([img_cat[0],img_cat[1],img_cat[2],img_cat[3]],1)
	torchvision.utils.save_image(img,"concat"+".jpg")

image = Image.open("COCO_TEST.jpg")
image2 = TF.center_crop(image,[128,128])
image_tensor = TF.to_tensor(image2)
image_tensor.unsqueeze_(0)
#print(image_tensor)
# Crop a 128 x 128 subimage in the top left corner
#cropped_image = image_tensor[0:1,0:3,0:128, 0:128]
imgs = cropImg(32,image_tensor)
concat_img = concatImgs(imgs)