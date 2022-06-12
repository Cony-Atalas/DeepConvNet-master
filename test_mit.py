import torch
import os
import sys
import cv2
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from Model import DeepNet
from torchvision import transforms
from scipy import ndimage
from PIL import Image
from load_imglist import ImageList
import PIL
import torchvision.transforms.functional as TF

global device
global count
count = 1
device = torch.device("cuda")


def find_scale_to_fit(img,shape):
    # finds the scale that makes the img fit in the rect
    w , h = img.shape[1] , img.shape[0]
    target_w , target_h = shape[1] , shape[0]
    scale = 1.0
    if target_w is not None:
        scale = min(scale,target_w/float(w))
    if target_h is not None:
        scale = min(scale,target_h/float(h))
    return scale


model = DeepNet()
model.load_state_dict(torch.load('./Models/Model_Save_new/DeepConvNet.pth.tar')['state_dict'])
model = model.to(device)
print('model loaded and test started')
model.eval()

file = open("./SaliconDataset/testMIT.txt")
for fp in file:

    print("Predict [{}/300]".format(count))

    fp = fp.replace('\n','')
    img = Image.open("./SaliconDataset/"+str(fp)).convert('RGB')

    img = np.array(img)
    # print(img.shape)
    h , w = img.shape[0] , img.shape[1]
    # print("h,w:",h,w)
    scale = find_scale_to_fit(img,(320,320))
    # print("scale",scale)
    img = Image.fromarray(img)
    img = TF.resize(img,[int(h*scale),int(w*scale)])
    # print("preprocess size:",img.size)
    img_tensor = transforms.ToTensor()
    img = img_tensor(img)
    # print("img_tensor",img.shape)
    img = img.unsqueeze(0)
    # print("img_tensor", img.shape)
    img = img.to(device)
    img_gt = model(img)
    # print("img_gt",img_gt.shape)
    # img_gt = img_gt.squeeze(0).squeeze(1)
    gt = img_gt.detach().cpu().numpy()
    gt = (gt-gt.min())/(gt.max()-gt.min())
    gt = gt*255
    gt = ndimage.gaussian_filter(gt,sigma=2)
    gt = gt.reshape(gt.shape[2],gt.shape[3])
    # print(gt.shape)
    # gt = transforms.ToPILImage()(np.uint8(gt),'L')
    # gt = Image.fromarray(gt)
    gt = Image.fromarray(np.uint8(gt),'L')
    gt = TF.resize(gt,(h,w))
    gt.save("./result/MIT/"+"out_{}.png".format(count))
    count = count+1
    # if count == 5:
    #     break
