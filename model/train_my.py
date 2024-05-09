from models import R50_UNET
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

import os

scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

logger = logging.getLogger()
fmt = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
file_handler = logging.FileHandler('train.log')
file_handler.setFormatter(fmt)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(fmt)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

train_dataset = dataset.ACDC('data/ACDC-2D-All/train/Img', "data/ACDC-2D-All/train/GT")
test_dataset = dataset.ACDC('data/ACDC-2D-All/val/Img', "data/ACDC-2D-All/val/GT")

model = R50_UNET.ResUnet(3)
now = 0
optimizer = torch.optim.AdamW(lr=3e-4, params=model.parameters())
criterion = FullyLoss(0.01, 0.01)

epochs = 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
loader = DataLoader(
        train_dataset,
        num_workers=15,
        batch_size=1,
        shuffle=True,
    )
loader_test = DataLoader(
        test_dataset,
        num_workers=15,
        batch_size=1,
        shuffle=False,
    )

def train(model, dataloader, loader_test, criterion, optimizer, scheduler, epochs):
    model.cuda()
    model.train()
    metric = Evaluator(2)
    max_mdice = 0
    max_epoch = 0
    for epoch in range(now+1, epochs + 1):
        start = time.time()
        metric.reset()
        losses = []
        for img, mask in tqdm(dataloader):
            img, mask = img.cuda(), mask.squeeze(1).cuda()
            optimizer.zero_grad()
            with autocast():
                preds, dist = model(img)
                loss = criterion(preds, dist, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pred_label = torch.argmax(preds, dim=1)
            pred_label = pred_label.squeeze(1).cpu().numpy()
            mask_ = mask.cpu().numpy()
            metric.add_batch(mask_, pred_label)
            losses.append(loss.item())
        scheduler.step()
        if epoch % 5 == 0:
            try:
                os.remove(f'pth/My-{epoch-10}.pth.tar')
            except:
                pass
            torch.save(model, f'pth/My-{epoch}.pth.tar')
        end = time.time()
        logger.info(
            f'train: epoch{epoch} loss:{sum(losses) / len(losses)} IOU:{metric.Mean_Intersection_over_Union()} DICE:{metric.Mean_DICE()} time:{end - start} MDICE:{np.mean(metric.Mean_DICE()[1:])} need time:{(epochs - epoch) * (end - start) / 3600}')
        max_mdice, max_epoch = test(model, loader_test, epoch, max_mdice, max_epoch)
        model.train()
        model.cuda()

def test(model, dataloader, epoch, max_mdice, max_epoch):
    model.cuda()
    model.eval()
    metric = Evaluator(2)
    for img, mask in tqdm(dataloader):
        img, mask = img.cuda(), mask.squeeze(1).long().cuda()
        with torch.no_grad():
            preds, _ = model(img)
        pred_label = torch.argmax(preds, dim=1)
        pred_label = pred_label.squeeze(1).cpu().numpy()
        mask_ = mask.cpu().numpy()
        metric.add_batch(mask_, pred_label)
    mdice = metric.Mean_DICE()
    t = np.mean(mdice[1:])
    logger.info(f'val: IOU:{metric.Mean_Intersection_over_Union()} DICE:{mdice} MDICE:{t}')
    if t > max_mdice:
        try:
            os.remove(f'pth/My-{max_epoch}-mdice:{max_mdice}.pth.tar')
        except:
            pass
        max_mdice = t
        max_epoch = epoch
        torch.save(model, f'pth/My-{epoch}-mdice:{max_mdice}.pth.tar')
    return max_mdice, max_epoch


train(model, loader, loader_test, criterion, optimizer, scheduler, epochs)