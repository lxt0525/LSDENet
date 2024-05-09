import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable


class DistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dist, mask):
        # dist: b class_num h w
        # mask: b h w
        b, c, h, w = dist.size()
        mask_one_hot = F.one_hot(mask.long(), num_classes=c)  # b h w c
        mask_one_hot = rearrange(mask_one_hot, 'b h w c-> b c h w', b=b, c=c, h=h, w=w)
        return dist[mask_one_hot == 1].pow(2).mean()
        # item = dist * (2 * mask_one_hot - 1)
        # return torch.mean(item)


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dist, mask):
        # dist: b class_num h w
        # mask: b h w
        return F.cross_entropy(-dist, mask.long())


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, mask, ep=1e-8):
        b, c, h, w = preds.size()
        mask_one_hot = F.one_hot(mask.long(), num_classes=c)  # b h w c
        mask_one_hot = rearrange(mask_one_hot, 'b h w c -> b c h w', b=b, c=c, h=h, w=w)
        preds = torch.softmax(preds, dim=1)
        preds = preds.view(b, c, -1)
        mask_one_hot = mask_one_hot.view(b, c, -1)
        intersection = 2 * torch.sum(preds * mask_one_hot, dim=2) + ep
        union = torch.sum(preds, dim=2) + torch.sum(mask_one_hot, dim=2) + ep
        dice = 1 - intersection / union
        return dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, weight=None, reduction='none', gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        b, c, h, w = input.size()
        logp = self.ce(input, target.long())
        mask_one_hot = F.one_hot(target.long(), num_classes=c)  # b h w c
        mask_one_hot = rearrange(mask_one_hot, 'b h w c -> b c h w', b=b, c=c, h=h, w=w)
        p = torch.exp(-logp)
        loss = ((1 - p) ** self.gamma) * logp
        return loss.mean()


class FullyLoss(nn.Module):
    def __init__(self, loss_dl_weight, loss_cl_weight):
        super().__init__()
        self.loss_dl_weight = loss_dl_weight
        self.loss_cl_weight = loss_cl_weight
        self.SegLoss = nn.CrossEntropyLoss(reduction='mean')
        self.DL = DistanceLoss()
        self.CL = ContrastiveLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, preds, dist, mask):

        return 0.4 * self.SegLoss(preds, mask.long()) + 0.6 * self.dice(preds, mask) + self.loss_dl_weight * self.DL(
            dist, mask) + self.loss_cl_weight * self.CL(dist, mask)


if __name__ == '__main__':
    x = torch.randn((3, 2, 256, 256))
    y = torch.zeros((3, 256, 256))
    print(FocalLoss()(x, y))






