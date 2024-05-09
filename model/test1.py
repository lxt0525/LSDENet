from models import REDE_V8
import torch
from configs import dataset, metric
from configs.metric import Evaluator
import time
from models.losses import FullyLoss
import numpy as np
import logging
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim
from PIL import Image
import os


test_dataset = dataset.ACDC_test('data/ACDC-2D-All/val/Img', 'data/ACDC-2D-All/val/GT')
# for i in range(2211):
#     img, mask, _ = test_dataset[i]
#     if len(mask.unique())==8:
#         print(i)




model = REDE_V8.Segmentor(3, 4)
model = torch.load('ACDC-V8-model-496-mdice_0.9310673774216256.pth.tar')


model.cuda()
model.eval()

item = 196
img, mask, _ = test_dataset[item]



with torch.no_grad():
    preds, _ = model(img.unsqueeze(0).cuda())

pred_label = torch.argmax(preds, dim=1)
pred_label = pred_label.squeeze()

img, img0 = test_dataset[item][2], test_dataset[item][2]
img = np.array(img)
img0 = np.array(img0)

mask = mask.squeeze().numpy()
pred_label = pred_label.cpu().numpy()
Image.fromarray(img).show()
img[np.where(mask == 1)] = np.array([0, 0, 255])
img[np.where(mask == 2)] = np.array([0, 255, 255])
img[np.where(mask == 3)] = np.array([255, 140, 0])

gt = Image.fromarray(img)
gt.show()
gt.save('gt.png')


img0[np.where(pred_label == 1)] = np.array([0, 0, 255])
img0[np.where(pred_label == 2)] = np.array([0, 255, 255])
img0[np.where(pred_label == 3)] = np.array([255, 140, 0])
pred = Image.fromarray(img0)
pred.show()
pred.save('pred.png')





