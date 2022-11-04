#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import torch
import torch.nn as nn



# In[13]:


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.act = act


    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        res = self.act(res)

        return res
# In[14]:

class CALayer2(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel*3, (channel*3) // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d((channel*3) // reduction,channel*3, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction,channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# In[15]:


class DCCL(nn.Module):
    """
    Sequential Dialted convolutions
    """

    def __init__(self, n_channels, n_dilations):
        super(DCCL, self).__init__()
        kernel_size = 3
        padding = 1
        #         self.dropout = nn.Dropout2d(dropout)
        self.conv1 = nn.Conv2d(n_channels * 3, n_channels, kernel_size=1)
        self.non_linearity = nn.ReLU(inplace=True)
        self.strides = [(2 * k) + 1 for k in range(n_dilations)]
        convs = [nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, dilation=s, padding=s) for s in
                 self.strides]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for c in convs:
            self.convs.append(c)
            self.bns.append(nn.BatchNorm2d(n_channels))

    def forward(self, x):
        skips = []
        for (c,bn,s) in zip(self.convs,self.bns,self.strides):
            x = c(x)
            x = bn(x)
            x = self.non_linearity(x)
            #             x = self.dropout(x)
            skips.append(x)
        x = torch.cat(skips, 1)
        # x = self.conv1(x)

        return x

class MAAM(nn.Module):
    """
    Sequential Dialted convolutions
    """

    def __init__(self, in_channels,n_channels):
        super(MAAM, self).__init__()
        self.inc = in_channels
        self.channels = n_channels
        self.conv1 = default_conv(self.inc,self.channels,1)
        self.conv3 = default_conv(self.inc,self.channels,3)
        self.conv5 = default_conv(self.inc, self.channels, 5)
        self.cat = default_conv(3*self.channels,self.channels,1)
        self.bypass = default_conv(self.inc, self.channels,1)
        self.sofx = nn.Softmax(dim=1)

    def forward(self, x):
        x_in = x
        z = self.bypass(x)
        x1 = self.conv1(x)
        x3= self.conv3(x)
        x5 = self.conv5(x)
        x = self.cat(torch.cat([x1,x3,x5],1))
        b = self.sofx(x)
        x = z * b
        att = x + x_in

        return att


# In[19]:
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = default_conv(n_feat, n_feat, 1, bias=bias)
        self.conv2 = default_conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = default_conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(default_conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(default_conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# In[20]:


class DCCA(nn.Module):
    """
    Sequential Dialted convolutions
    """

    def __init__(self, n_channels=32, n_dilations=3, kernel_size=3, reduction=4, bias=False,act=nn.PReLU()):
        super(DCCA, self).__init__()
        self.channels = n_channels
        self.n_dilations = n_dilations
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.act = act
        self.conv1 = nn.Conv2d(n_channels * 3, n_channels, kernel_size=1)
        self.conv = default_conv(self.channels, self.channels, 3)
        self.DCCL = DCCL(self.channels, self.n_dilations)
        self.CA = CALayer2(self.channels, self.reduction)
        # if bias:
        #     self.CA = CALayer2(self.channels, self.reduction,bias=True)
        self.SAM = SAM(self.channels, self.kernel_size, bias=False)
        self.norm = nn.BatchNorm2d(self.channels)

    def forward(self, x):
        x_in = x

        x = self.conv(x)

        x = self.act(x)
        x = self.DCCL(x)

        #         x = self.norm(x)
        x = self.CA(x)
        x = self.conv1(x)

        x = x_in + x

        return x


# In[21]:


class DCANet(nn.Module):
    """
    Sequential Dialted convolutions
    """

    def __init__(self, n_channels, n_dilations, kernel_size, reduction, act):
        super(DCANet, self).__init__()
        self.channels = n_channels
        self.n_dilations = n_dilations
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.act = act
        self.conv_0 = default_conv(3, self.channels, self.kernel_size)
        self.conv = default_conv(self.channels, self.channels, self.kernel_size)
        self.conv_end = default_conv(self.channels, 3, self.kernel_size)
        self.DCCA1 = DCCA(self.channels, self.n_dilations, self.kernel_size, self.reduction, self.act)
        self.DCCA2 = DCCA(self.channels, self.n_dilations, self.kernel_size, self.reduction, self.act)
        self.DCCA3 = DCCA(self.channels, self.n_dilations, self.kernel_size, self.reduction, self.act)
        self.DCCA4 = DCCA(self.channels, self.n_dilations, self.kernel_size, self.reduction, self.act)
        self.DCCA5 = DCCA(self.channels, self.n_dilations, self.kernel_size, self.reduction, self.act)
        self.DCCA6 = DCCA(self.channels, self.n_dilations, self.kernel_size, self.reduction, self.act)
        self.DCCA7 = DCCA(self.channels, self.n_dilations, self.kernel_size, self.reduction, self.act)
        # self.DCCA8 = DCCA(self.channels, self.n_dilations, self.kernel_size, self.reduction, self.act)
        # self.DCCL1 = DCCL(self.channels, self.n_dilations)
        # self.DCCL2 = DCCL(self.channels, self.n_dilations)
        # self.DCCL3 = DCCL(self.channels, self.n_dilations)
        # self.DCCL3 = DCCL(self.channels, self.n_dilations)
        # self.DCCL5 = DCCL(self.channels, self.n_dilations)
        # self.DCCL6 = DCCL(self.channels, self.n_dilations)
        # self.DCCL7 = DCCL(self.channels, self.n_dilations)
        self.CAB = CAB(self.channels, self.kernel_size, self.reduction, act=self.act, bias=False)
        # self.RB = ResBlock(default_conv,self.channels,self.kernel_size)
        self.SAM = SAM(self.channels, self.kernel_size, bias=False)
        # self.norm = nn.BatchNorm2d(self.channels)
        # self.maam = MAAM(3,self.channels)

    def forward(self, x):
        x_0 = x
        # att = self.maam(x)
        x = self.conv_0(x)
        # x = self.RB(x)
        x = self.CAB(x)
        x_in = x
        ##DCCA Block
        x = self.DCCA1(x)
        x = self.DCCA2(x)
        x = self.DCCA3(x)
        x = self.DCCA4(x)
        x = self.DCCA5(x)
        x = self.DCCA6(x)
        x = self.DCCA7(x)
        # x = self.DCCA8(x)

        # x = self.DCCL1(x)
        # x = self.DCCL2(x)
        # x = self.DCCL3(x)
        # x = self.DCCL4(x)
        # x = self.DCCL5(x)
        # x = self.DCCL6(x)
        # x = self.DCCL7(x)
        #         x = self.norm(x)
        x = x_in + x
        # x = self.conv_end(x)
        res, img = self.SAM(x, x_0)
        rain = self.conv_end(res)
        return rain, img

# In[ ]:





# In[ ]:




