import os
import sys

import torch
import torch.nn.functional as F
from torchjpeg import dct
import numpy as np

def cosine_similarity(x,w):
    x_norm = F.normalize(x,dim=1)
    w_norm = F.normalize(w,dim=1)
    cosine = torch.mm(x_norm, w_norm.T).clamp(-1, 1)
    cosine=(cosine+1)/2
    return cosine

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')



def _dct_to_images(x, size=8, stride=8, pad=0, dilation=1):
    bs, _, _, _ = x.shape
    sampling_rate = 8

    x = x.view(bs, 3, 64, 14 * sampling_rate, 14 * sampling_rate)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(bs, 3, 14 * 14 * sampling_rate * sampling_rate, 8, 8)
    x = dct.block_idct(x)
    x = x.view(bs * 3, 14 * 14 * sampling_rate * sampling_rate, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(112 * sampling_rate, 112 * sampling_rate),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(bs, 3, 112 * sampling_rate, 112 * sampling_rate)
    x += 128
    x = dct.to_rgb(x)
    x /= 255
    x = F.interpolate(x, scale_factor=1 / sampling_rate, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x



def _dct_to_images2(x, size=8, stride=8, pad=0, dilation=1):
    bs, _, _, _ = x.shape
    sampling_rate = 8

    x = x.view(bs, 3, 64, 14 * sampling_rate, 14 * sampling_rate)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(bs, 3, 14 * 14 * sampling_rate * sampling_rate, 8, 8)
    x = dct.block_idct(x)
    x = x.view(bs * 3, 14 * 14 * sampling_rate * sampling_rate, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(112 * sampling_rate, 112 * sampling_rate),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(bs, 3, 112 * sampling_rate, 112 * sampling_rate)
    x += 128
    x = dct.to_rgb(x)
    x /= 255
    x = F.interpolate(x, scale_factor=1 / sampling_rate, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x
