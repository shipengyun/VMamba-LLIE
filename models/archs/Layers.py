
import math

import torch.nn as nn
import torch
from torch.nn import GELU
import torch.nn.functional as F
from models.archs.SS2D_arch import SS2D
from models.archs.SubLayers import MultiHeadAttention,ShuffleAttention,CBAMLayer



class PreNorm(nn.Module):
    """
    预归一化模块，通常用于Transformer架构中。

    在执行具体的功能（如自注意力或前馈网络）之前先进行层归一化，
    这有助于稳定训练过程并提高模型性能。

    属性:
        dim: 输入特征的维度。
        fn: 要在归一化后应用的模块或函数。
    """

    def __init__(self, dim, fn,dropout=0.1):
        """
        初始化预归一化模块。

        参数:
            dim (int): 输入特征的维度，也是层归一化的维度。
            fn (callable): 在归一化之后应用的模块或函数。
        """
        super().__init__()  # 初始化基类 nn.Module
        self.fn = fn  # 存储要应用的函数或模块
        self.norm = nn.LayerNorm(dim)  # 创建层归一化模块
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        """
        对输入数据进行前向传播。

        参数:
            x (Tensor): 输入到模块的数据。
            *args, **kwargs: 传递给self.fn的额外参数。

        返回:
            Tensor: self.fn的输出，其输入是归一化后的x。
        """
        x = self.norm(x)  # 首先对输入x进行层归一化
        return self.fn(x, *args, **kwargs)  # 将归一化的数据传递给self.fn，并执行
class FeedForward(nn.Module):
    """
    实现一个基于卷积的前馈网络模块，通常用于视觉Transformer结构中。
    这个模块使用1x1卷积扩展特征维度，然后通过3x3卷积在这个扩展的维度上进行处理，最后使用1x1卷积将特征维度降回原来的大小。

    参数:
        dim (int): 输入和输出特征的维度。
        mult (int): 特征维度扩展的倍数，默认为4。
    """

    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),  # 使用1x1卷积提升特征维度
            GELU(),  # 使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),  # 分组卷积处理，维持特征维度不变，增加特征的局部相关性
            GELU(),  # 再次使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),  # 使用1x1卷积降低特征维度回到原始大小
        )

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (tensor): 输入特征，形状为 [b, h, w, c]，其中b是批次大小，h和w是空间维度，c是通道数。

        返回:
        out (tensor): 输出特征，形状与输入相同。
        """
        # 由于PyTorch的卷积期望的输入形状为[b, c, h, w]，需要将通道数从最后一个维度移到第二个维度
        out = self.net(x.permute(0, 3, 1, 2).contiguous())  # 调整输入张量的维度
        return out.permute(0, 2, 3, 1)  # 将输出张量的维度调整回[b, h, w, c]格式



class EncoderLayer3(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer3, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_model,d_inner,dropout=dropout) #dropout=dropout
        self.sa = ShuffleAttention(d_model)
        self.ss2d = SS2D(d_model=d_model, dropout=0, d_state=16)
        self.ffn = PreNorm(d_model, FeedForward(dim=d_model),dropout=0.1)
        self.gelu = nn.GELU()
        self.c = CBAMLayer(d_model)

    def forward(self, enc_input, slf_attn_snr=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input,slf_attn_snr)
        enc_output = enc_output.permute(0, 2, 3, 1)
        slf_attn_snr= slf_attn_snr.permute(0, 2, 3, 1)
        x = self.ss2d(enc_output,slf_attn_snr) + enc_input.permute(0, 2, 3, 1)
        x = self.ffn(x) + x
        enc_output = x.permute(0, 3, 1, 2)   
        enc_output1=self.c(enc_input)
        enc_output1=self.sa(enc_output1)
        enc_output = enc_output+enc_output1[0]

        return enc_output, enc_slf_attn

