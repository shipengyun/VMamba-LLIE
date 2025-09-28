import math
import torch.nn as nn
import torch
from torch.nn import GELU
import torch.nn.functional as F
from models.archs.SS2D_arch import SS2D
from models.archs.SubLayers import MultiHeadAttention, CAA,ShuffleAttention,ELAB,CBAMLayer



class PreNorm(nn.Module):
    def __init__(self, dim, fn,dropout=0.1):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):

        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)



class EncoderLayer3(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer3, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_model,d_inner,dropout=dropout) #dropout=dropout
        # self.sa = ShuffleAttention(d_model)
        self.ss2d = SS2D(d_model=d_model, dropout=0, d_state=16)
        self.ffn = PreNorm(d_model, FeedForward(dim=d_model),dropout=0.1)
        self.gelu = nn.GELU()
        # self.c = CBAMLayer(d_model)

    def forward(self, enc_input, slf_attn_snr=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input,slf_attn_snr)
        x = self.ss2d(enc_output.permute(0, 2, 3, 1),slf_attn_snr.permute(0, 2, 3, 1)) + enc_input.permute(0, 2, 3, 1)
        x = self.ffn(x) + x
        enc_output = x.permute(0, 3, 1, 2)   
        # enc_output1=self.c(enc_input)
        # enc_output1=self.sa(enc_output1)
        enc_output = enc_output+enc_output1[0]

        return enc_output, enc_slf_attn

