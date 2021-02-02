import torch
import torch.nn as nn
from einops import rearrange

class CC(nn.Module):
    def __init__(self,  kernel):
        super(CC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=int((kernel - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of CC module
        t = rearrange(rearrange(y, 'b c w h->b c (w h)'), 'b c w->b w c')
        y = self.conv(t)
        y = rearrange(rearrange(y, 'b c w->b w c'), 'b c (w h)->b c w h', w=1, h=1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SC(nn.Module):
    def __init__(self):
        super(SC, self).__init__()
        kernel_size = 3
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class CCWH(nn.Module):
    def __init__(self):
        super(CCWH, self).__init__()
        self.C_H = SC()
        self.C_W = SC()
        self.C_C = CC(3)

    def forward(self, x):
        x_perm1 = rearrange(x, 'b c w h->b w c h')
        x_out1 = self.C_H(x_perm1)
        x_out11 = rearrange(x_out1, 'b c w h->b w c h').contiguous()

        x_perm2 = rearrange(x, 'b c w h->b h w c').contiguous()
        x_out2 = self.C_W(x_perm2)
        x_out21 = rearrange(x_out2, 'b c w h->b h w c').contiguous()
        x_out = self.C_C(x)
        x_out = (1 / 2) * (x_out11 + x_out21) + x_out
        return x_out
# model = CCWH()
# import time
# input = torch.randn(1, 160, 40,40)
# start=time.time()
# out = model(input)
# print(time.time()-start)
# print(out.shape)
