import torch
import torch.nn as nn
from einops import rearrange
# class CC(nn.Module):
#     """Constructs a C_C module.
#     Args:
#         channel: Number of channels of the input feature map
#     """
#     def __init__(self, kernel):
#         super(CC, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=int((kernel-1)//2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#         # Two different branches of CC module
#         t=rearrange(rearrange(y,'b c w h->b c (w h)'),'b c w->b w c')
#         y = self.conv(t)
#         y=rearrange(rearrange(y,'b c w->b w c'),'b c (w h)->b c w h',w=1,h=1)
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)
class CC(nn.Module):
    def __init__(self, channel,k_size):
        super(CC, self).__init__()
        self.k_size=k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 3
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out
class CCWH(nn.Module):
    def __init__(self, channels):
        super(CCWH, self).__init__()
        self.C_H = AttentionGate()
        self.C_W = AttentionGate()
        self.C_C=CC(channels,3)
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.C_W(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.C_H(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        x_out = self.C_C(x)
        #print(x_out11.shape,x_out.shape)
        x_out = (1/2)*(x_out11 + x_out21)+x_out
        return x_out
# model = CCWH(160)
# import time
# input = torch.randn(4, 160, 32,32)
# start=time.time()
# out = model(input)
# print(time.time()-start)
# print(out.shape)