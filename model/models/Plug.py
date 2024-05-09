import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
import math


class Description(nn.Module):
    def __init__(self, channel, space):
        super(Description, self).__init__()

        self.weight = nn.Parameter(torch.randn(channel, space))
        trunc_normal_(self.weight, std=0.02)

    def forward(self):
        # input: (bs*t) x dim
        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight)
        return Q


class Enhance(nn.Module):
    def __init__(self, channel, dim):
        super().__init__()
        self.description_global = Description(channel, dim)
        self.description_channel = Description(channel, dim)
        self.cnn_global = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cnn_channel = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: b c h w
        b, c, h, w = x.size()
        # global
        g = self.cnn_global(x)
        g = rearrange(g, 'b c h w -> (b h w) c', b=b, h=1, w=1)
        Q = self.description_global()
        affinity_global = torch.softmax(torch.matmul(g, Q) / math.sqrt(c), dim=1)
        g = torch.matmul(affinity_global, Q.T)
        g = torch.sigmoid(g)
        # channel
        ch = self.cnn_channel(x)
        ch = rearrange(ch, 'b c h w -> b (h w) c', b=b, h=h, w=w)
        affinity_channel = torch.softmax(torch.matmul(ch, Q) / math.sqrt(c), dim=2)
        ch = torch.sum(torch.diag_embed(affinity_channel) @ Q.T, dim=2)
        o = ch * g[:, None, :]
        o = rearrange(o, 'b (h w) c -> b c h w', b=b, h=h, w=w)
        return o + x


class ReEmbedding(nn.Module):
    def __init__(self, channel, num_classes):
        super().__init__()
        self.channel = channel
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(channel*num_classes))
        self.ebd = nn.Sequential(
            nn.Conv2d(channel, channel*num_classes, kernel_size=1, bias=False)

        )

    def forward(self, x):
        # x: b c h w
        b, c, h, w = x.size()
        # x = rearrange(x, 'b c h w -> (b h w) c', b=b, h=h, w=w)
        x = self.ebd(x)
        # x = rearrange(x, '(b h w) c -> b c h w', b=b, h=h, w=w)

        t = torch.softmax(self.weight.view((self.channel, self.num_classes)), dim=0).view(-1)
        x = x.pow(2)  # b c*n h w
        x = rearrange(x, 'b (c n) h w -> (b h w) (c n)', b=b, c=c, h=h, w=w)
        x = x * t
        x = rearrange(x, '(b h w) (c n) -> b (c n) h w ', b=b, c=c, h=h, w=w)
        dist = []
        for i in range(self.num_classes):
            dist.append(torch.sqrt(torch.sum(x[:, (i*self.channel):((i+1)*self.channel), :, :], dim=1).unsqueeze(1)))
        dist = torch.cat(dist, dim=1)
        return dist