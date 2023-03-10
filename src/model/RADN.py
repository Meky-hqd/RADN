from model.ref_model.Restormer import *
from model.ref_model.DCCA_x2 import *
from torch.nn import init
from torch.autograd import Variable
import torch.nn as nn
import torch

N_CHANNELS = 32


class ConvLSTM(nn.Module):

    def __init__(self, nb_channel, softsign):
        super(ConvLSTM, self).__init__()

        self.conv_i = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        self.conv_f = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        self.conv_g = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        self.conv_o = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)

        init.orthogonal(self.conv_i.weight)
        init.orthogonal(self.conv_f.weight)
        init.orthogonal(self.conv_g.weight)
        init.orthogonal(self.conv_o.weight)

        init.constant(self.conv_i.bias, 0.)
        init.constant(self.conv_f.bias, 0.)
        init.constant(self.conv_g.bias, 0.)
        init.constant(self.conv_o.bias, 0.)

        self.conv_ii = nn.Sequential(self.conv_i, nn.Sigmoid())
        self.conv_ff = nn.Sequential(self.conv_f, nn.Sigmoid())
        if not softsign:
            self.conv_gg = nn.Sequential(self.conv_g, nn.Tanh())
        else:
            self.conv_gg = nn.Sequential(self.conv_g, nn.Softsign())

        self.conv_oo = nn.Sequential(self.conv_o, nn.Sigmoid())

        self.nb_channel = nb_channel

    def forward(self, input, prev_h, prev_c):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        if prev_h is None:
            prev_h = Variable(torch.zeros(batch_size, self.nb_channel, row, col)).cuda()
            prev_c = Variable(torch.zeros(batch_size, self.nb_channel, row, col)).cuda()

        x = torch.cat((input, prev_h), 1)
        i = self.conv_ii(x)
        f = self.conv_ff(x)
        g = self.conv_gg(x)
        o = self.conv_oo(x)
        c = f * prev_c + i * g
        h = o * torch.tanh(c)

        return h, c


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class FAM(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(FAM, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class RRU(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, kernel=3, h_dim=32, ):
        super(RRU, self).__init__()
        # self.con0 = default_conv(3,2*n_channels,3)
        self.convin = default_conv(n_channels, n_channels, kernel)
        self.act1 = nn.PReLU()
        self.lstm = ConvLSTM(32, False)
        self.convout = default_conv(n_channels, n_channels, kernel)
        # self.conv_img = default_conv(n_channels,3,3)
        self.act2 = nn.PReLU()

    def forward(self, x, h, c):
        x_in = x
        x = self.act1(self.convin(x))
        h, c = self.lstm(x, h, c)
        # print('lstmout_shape',x.shape)
        x = h
        out = self.act2(self.convout(x))
        out = x_in + out
        # print('final_shape', x.shape)
        # out = self.conv_img(out)

        return out, h, c


class RRU_D(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, kernel=3, h_dim=32, ):
        super(RRU_D, self).__init__()
        # self.con0 = default_conv(3,2*n_channels,3)
        self.convin = default_conv(n_channels, n_channels, kernel)
        self.act1 = nn.PReLU()
        self.lstm = ConvLSTM(32, False)
        self.fam = FAM(default_conv, 32, 3)
        self.convout = default_conv(n_channels, n_channels, kernel)
        # self.conv_img = default_conv(n_channels,3,3)
        self.act2 = nn.PReLU()

    def forward(self, x, h, c):
        x_in = x
        x = self.act1(self.convin(x))
        x = self.fam(x)
        h, c = self.lstm(x, h, c)
        # print('lstmout_shape',x.shape)
        x = h
        out = self.act2(self.convout(x))
        out = x_in + out
        # print('final_shape', x.shape)
        # out = self.conv_img(out)

        return out, h, c


class RRU_Test(nn.Module):
    def __init__(self, rru_nums, n_channels=N_CHANNELS, kernel=3):
        super(RRU_Test, self).__init__()
        self.rru_nums = rru_nums
        self.convin = default_conv(3, n_channels, kernel)
        self.rru = RRU_D()
        self.trm = TransformerBlock(dim=n_channels, num_heads=8, ffn_expansion_factor=2.66, bias=False,
                                    LayerNorm_type="WithBias")
        self.conv_f = default_conv(n_channels, n_channels, kernel)
        self.conv_img = default_conv(n_channels, 3, kernel)

    def forward(self, x):
        x_in = x
        h = None
        c = None
        x = self.convin(x)
        for i in range(self.rru_nums):
            x, h, c = self.rru(x, h, c)
            x = self.trm(x)
        x = self.conv_f(x)
        out = self.conv_img(x)
        out = x_in + out

        return out


class CFM(nn.Module):
    def __init__(self, batch_size, rru_nums, n_channels=N_CHANNELS, kernel=3):
        super(CFM, self).__init__()
        self.batch_size = batch_size
        self.rru_nums = rru_nums
        # self.convin = default_conv(3, n_channels, kernel)
        self.rrus = nn.ModuleList()
        self.trm = nn.ModuleList()
        for i in range(rru_nums):
            self.rrus.append(RRU(self.batch_size))
            self.trm.append(TransformerBlock(dim=n_channels, num_heads=8, ffn_expansion_factor=2.66, bias=False,
                                             LayerNorm_type="WithBias"))
        self.conv_f = default_conv(n_channels, n_channels, kernel)
        # self.conv_img = default_conv(n_channels,3,kernel)

    def forward(self, x):
        x_in = x
        # x = self.convin(x)
        for rru, trm in zip(self.rrus, self.trm):
            x = rru(x)
            x = trm(x)
        out = self.conv_f(x)
        # out = self.conv_img(x)
        out = x_in + out

        return out


class Basic_blocks(nn.Module):
    def __init__(self, batch_size, rru_nums, n_channels=N_CHANNELS, kernel=3):
        super(Basic_blocks, self).__init__()
        self.batch_size = batch_size
        self.rru_nums = rru_nums

        self.rrus = nn.ModuleList()
        self.trm = nn.ModuleList()
        for i in range(rru_nums):
            self.rrus.append(RRU(self.batch_size))
            self.trm.append(TransformerBlock(dim=n_channels, num_heads=8, ffn_expansion_factor=2.66, bias=False,
                                             LayerNorm_type="WithBias"))

    def forward(self, x):
        x_in = x
        for rru, trm in zip(self.rrus, self.trm):
            x = rru(x)
            x = trm(x)
        out = x_in + x

        return out


class Dialated(nn.Module):
    """
    Sequential Dialted convolutions
    """

    def __init__(self, n_channels, n_dilations):
        super(Dialated, self).__init__()
        kernel_size = 3
        padding = 1
        #         self.dropout = nn.Dropout2d(dropout)
        self.conv1 = nn.Conv2d(n_channels * n_dilations, n_channels, kernel_size=1)
        self.non_linearity = nn.ReLU(inplace=True)
        self.strides = [(2 * k) + 1 for k in range(n_dilations)]
        convs = [nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, dilation=s, padding=s) for s in
                 self.strides]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for c in convs:
            self.convs.append(c)
            self.bns.append(LayerNorm(n_channels, LayerNorm_type="WithBias"))

    def forward(self, x):
        skips = []
        for (c, bn, s) in zip(self.convs, self.bns, self.strides):
            x = c(x)
            x = bn(x)
            x = self.non_linearity(x)
            #             x = self.dropout(x)
            skips.append(x)
        x = torch.cat(skips, 1)
        # print("dia_cat:",x.shape)

        x = self.conv1(x)

        return x


class TRM_D_Block(nn.Module):
    def __init__(self, f_dim=N_CHANNELS, n_heads=N_CHANNELS // 8, n_dials=1):
        super(TRM_D_Block, self).__init__()
        self.n_heads = n_heads
        self.trm = TransformerBlock(dim=f_dim, num_heads=self.n_heads, ffn_expansion_factor=2.66, bias=False,
                                    LayerNorm_type="WithBias")
        self.dia = Dialated(f_dim, n_dials)

    def forward(self, x):
        x_in = x
        x = self.trm(x)
        x = self.dia(x)

        out = x + x_in
        return out



class Recurrent_T(nn.Module):
    def __init__(self, rru_nums, n_channels=N_CHANNELS, kernel=3):
        super(Recurrent_T, self).__init__()
        self.rru_nums = rru_nums
        self.convin = default_conv(3, n_channels, kernel)
        self.rru = RRU_D()
        self.trm = TransformerBlock(dim=n_channels, num_heads=8, ffn_expansion_factor=2.66, bias=False,
                                    LayerNorm_type="WithBias")
        self.conv_f = default_conv(n_channels, n_channels, kernel)
        self.conv_img = default_conv(n_channels, 3, kernel)

    def forward(self, x):
        x_in = x
        h = None
        c = None
        feature_list = []
        x = self.convin(x)
        for i in range(self.rru_nums):
            x, h, c = self.rru(x, h, c)
            x = self.trm(x)
            feature_list.append(x)
        x = self.conv_f(x)
        out = self.conv_img(x)
        # out = x_in + out

        return out, feature_list

class Recurrent_S(nn.Module):
    def __init__(self, rru_nums, n_channels=N_CHANNELS, kernel=3):
        super(Recurrent_S, self).__init__()
        self.rru_nums = rru_nums
        self.convin = default_conv(3, n_channels, kernel)
        self.rru = RRU_D()
        self.trm = TransformerBlock(dim=n_channels, num_heads=8, ffn_expansion_factor=2.66, bias=False,
                                    LayerNorm_type="WithBias")
        self.conv_f = default_conv(n_channels, n_channels, kernel)
        self.conv_img = default_conv(n_channels, 3, kernel)

    def forward(self, x):
        x_in = x
        h = None
        c = None
        feature_list = []
        x = self.convin(x)
        for i in range(self.rru_nums):
            x, h, c = self.rru(x, h, c)
            x = self.trm(x)
            feature_list.append(x)
        x = self.conv_f(x)
        out = self.conv_img(x)
        out = x_in + out

        return out, feature_list


if __name__ == '__main__':
    x = torch.randn(1, 3, 96, 96).cuda()
    m = Recurrent_S(2).cuda()
    y,_ = m(x)
    print(y.shape)