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


test_dataset = dataset.Synapse_test('data/train')
# for i in range(2211):
#     img, mask, _ = test_dataset[i]
#     if len(mask.unique())==8:
#         print(i)




model = REDE_V8.Segmentor(3, 9)
model = torch.load('Synapse-V8-model-131-mdice_0.8625396059460171.pth.tar')


model.cuda()
model.eval()
# # 425
# # 731
# # 732
# # 733
# # 734
# # 735
# # 736
# # 737
# # 738
# # 833
# # 834
# # 835
# # 836
# # 837
# # 838
# # 839
# # 840
# 1144
# 1145
# 1146
# 1147
# 1148
# 1149
# 1150
# 1151
# 1152
# 1153
# 1410
# 1411
# 1412
# 1413
# 1414
# 1415
# 1416
# 1417
# 1418
# 1537
# 1538
# 1539 *
# 1540
#
#
#
#
#
item = 491
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
img[np.where(mask == 4)] = np.array([34, 139, 34])
img[np.where(mask == 5)] = np.array([255, 0, 255])
img[np.where(mask == 6)] = np.array([128, 0, 128])
img[np.where(mask == 7)] = np.array([255, 0, 0])
img[np.where(mask == 8)] = np.array([255, 255, 0])
gt = Image.fromarray(img)
gt.show()
gt.save('gt.png')


img0[np.where(pred_label == 1)] = np.array([0, 0, 255])
img0[np.where(pred_label == 2)] = np.array([0, 255, 255])
img0[np.where(pred_label == 3)] = np.array([255, 140, 0])
img0[np.where(pred_label == 4)] = np.array([34, 139, 34])
img0[np.where(pred_label == 5)] = np.array([255, 0, 255])
img0[np.where(pred_label == 6)] = np.array([128, 0, 128])
img0[np.where(pred_label == 7)] = np.array([255, 0, 0])
img0[np.where(pred_label == 8)] = np.array([255, 255, 0])
pred = Image.fromarray(img0)
pred.show()
pred.save('pred.png')





