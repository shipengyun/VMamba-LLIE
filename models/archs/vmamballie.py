import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import models.archs.arch_util as arch_util
import numpy as np
import cv2
from models.archs.Models import Encoder_patch66
from models.archs.HVI_CAB.hvi import hvi

###############################
class vmamballie(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=True, w_TSA=True):
        super(vmamballie, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        ResidualBlock_noBN_f1 = functools.partial(arch_util.ResidualBlock_noBN1, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)


        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.snrvmamba = Encoder_patch66(d_model=64, d_inner=128, n_layers=6)
        self.lder = arch_util.make_layer(ResidualBlock_noBN_f1, 6)
        self.side_out = nn.Conv2d(nf, 3, 3, stride=1, padding=1)

        self.hvi =hvi()

    def forward(self, x, snr=None):

        x_center = x  # x:[b,c,h,w]
        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))  # [b,nf,h,w]
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))  # [b,nf,h/2,w/2]
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))  # [b,nf,h/4,w/4]

        fea = self.feature_extraction(L1_fea_3)  # [b,nf,h/4,w/4]
        fea_light = self.lder(fea)  # [b,nf,h/4,w/4]
        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        snr = F.interpolate(snr, size=[h_feature, w_feature], mode='nearest')

        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)

        fea_unfold = fea
        snr_unfold = snr
        fea_unfold = self.snrvmamba(fea_unfold, xs, src_snr=snr_unfold)

        channel = fea.shape[1]
        snr = snr.repeat(1, channel, 1, 1)  # [b,nf,h/4,w/4]


        fea =fea_light * (snr)+ fea_unfold * (1 - snr)
        out_fea = self.side_out(fea)
        out_noise = self.recon_trunk(fea)  #  [b,nf,h/4,w/4]

        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))

        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))

        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))

        out_noise = self.conv_last(out_noise)


        out_noise = out_noise + x_center
        out_noise = self.hvi(x) + out_noise

        return out_noise,out_fea
