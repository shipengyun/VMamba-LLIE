import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.archs.HVI_CAB.HVI_transform import RGB_HVI
from models.archs.HVI_CAB.utils import *
from models.archs.HVI_CAB.BCAVMamba import *
from models.archs.SS2D_arch import SS2D


class hvi(nn.Module):
    def __init__(self,
                 channels=[64, 64, 128, 128],
                 heads=[1, 2, 4, 8],
                 norm=False
                 ):
        super(hvi, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
       
      
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)


    
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_BCA1 = HV_BCA(ch2, head2)


        self.HV_BCA3 = HV_BCA(ch4, head4)
        self.HV_BCA4 = HV_BCA(ch4, head4)
        self.HV_BCA6 = HV_BCA(ch2, head2)

        self.ss2d1 = SS2D(ch2)
        self.ss2d3 = SS2D(ch4)
        self.ss2d4 = SS2D(ch4)
        self.ss2d6 = SS2D(ch2)

        self.I_BCA1 = I_BCA(ch2, head2)
      
        self.I_BCA3 = I_BCA(ch4, head4)
        self.I_BCA4 = I_BCA(ch4, head4)
       
        self.I_BCA6 = I_BCA(ch2, head2)

        self.trans = RGB_HVI().cuda()

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_BCA1(i_enc1, hv_1)
        hv_2 = self.HV_BCA1(hv_1, i_enc1)
        # print(hv_2.shape)
        hv_2 = self.ss2d1(hv_2.permute(0, 2, 3, 1), None).permute(0, 3, 1, 2)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)


        i_enc4 = self.I_BCA3(i_enc2, hv_2)
        hv_4 = self.HV_BCA3(hv_2, i_enc2)
        # print(hv_4.shape)
        hv_4 = self.ss2d3(hv_4.permute(0, 2, 3, 1), None).permute(0, 3, 1, 2)

        i_dec4 = self.I_BCA4(i_enc4, hv_4)
        hv_4 = self.HV_BCA4(hv_4, i_enc4)  # 中间层
        hv_4 = self.ss2d4(hv_4.permute(0, 2, 3, 1), None).permute(0, 3, 1, 2)


        hv_2 = self.HVD_block2(hv_4, hv_jump1)
        i_dec2 = self.ID_block2(i_dec4, v_jump1)
        i_dec1 = self.I_BCA6(i_dec2, hv_2)
        hv_1 = self.HV_BCA6(hv_2, i_dec2)
        hv_1 = self.ss2d6(hv_1.permute(0, 2, 3, 1), None).permute(0, 3, 1, 2)


        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)


        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi



