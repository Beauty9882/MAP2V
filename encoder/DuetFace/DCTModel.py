import torch
import sys
import torch.nn as nn

import torch
import torch.nn.functional as F
from torchjpeg import dct
import numpy as np

class DCTModel(nn.Module):
    def __init__(self, sub_channels = [ 0, 1, 2, 3, 4, 5, 8, 9, 16, 24 ]):
        super(DCTModel, self).__init__()
        self.sub_channels = sub_channels

    def _images_to_dct(self,x, sub_channels=None, size=8, stride=8, pad=0, dilation=1):
        x = x * 0.5 + 0.5  # x [-1, 1]  to [0, 1]

        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)#使用双线性插值将输入图像 x 的尺寸放大到原始尺寸的 8 倍。
        x *= 255 #将输入图像 x 的值从范围 [0, 1] 缩放到范围 [0, 255]
        if x.shape[1] == 3:#如果输入图像 x 的通道数为 3，则将其转换为 YCbCr 颜色空间。
            x = dct.to_ycbcr(x)
        x -= 128  # x to [-128, 127]
        bs, ch, h, w = x.shape#获取输入图像 x 的批次大小（batch size）、通道数（channels）、高度（height）和宽度（width）
        block_num = h // stride#根据给定的步幅 stride 计算图像高度上的块数
        x = x.view(bs * ch, 1, h, w)#将输入图像 x 的形状重塑为 (bs * ch, 1, h, w)
        x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad,
                    stride=(stride, stride))#将输入图像 x 进行图像块展开操作，使用给定的卷积核大小、膨胀率、填充和步幅。
        x = x.transpose(1, 2)#交换张量维度，将展开的图像块维度放置在正确的位置。
        x = x.view(bs, ch, -1, size, size)#将展开的图像块维度重塑
        dct_block = dct.block_dct(x)#对展开的图像块进行分块 DCT 变换。
        dct_block = dct_block.view(bs, ch, block_num, block_num, size * size).permute(0, 1, 4, 2, 3)#将分块 DCT 变换后的结果重塑为 (bs, ch, size^2, block_num, block_num) 并进行维度置换

        channels = list(set([i for i in range(64)]) - set(sub_channels))
        main_inputs = dct_block[:, :, channels, :, :]
        sub_inputs = dct_block[:, :, sub_channels, :, :]
        main_inputs = main_inputs.reshape(bs, -1, block_num, block_num)
        sub_inputs = sub_inputs.reshape(bs, -1, block_num, block_num)
        return main_inputs, sub_inputs

    def forward(self, x):
        x_server, _  = self._images_to_dct(x, sub_channels=self.sub_channels)#main_inputs:[30,162,112,112] sub_inputs:[30,30,112,112]
        return x_server

def _dct_to_images(x, size=8, stride=8, pad=0, dilation=1):
    bs, _, _, _ = x.shape
    sampling_rate = 8

    x = x.view(bs, 3, 54, 14 * sampling_rate, 14 * sampling_rate)
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

if __name__ == "__main__":
    mymodel = DCTModel()
    image_size1 = (1,3,112, 112)  # Width, Height, Channels (RGB)
    image_size = (4,3,112, 112)  # Width, Height, Channels (RGB)

    # Generate random pixel values between 0 and 1
    random_image = torch.rand(*image_size1)
    # If you want to convert it to a valid image with values in [0, 255],
    # you can multiply by 255 and convert to integers
    # random_image *= 255
    # random_image = random_image.int()

    # Generate random pixel values between 0 and 1
    random_image2 = torch.rand(*image_size)
    # If you want to convert it to a valid image with values in [0, 255],
    # you can multiply by 255 and convert to integers
    # random_image2 *= 255
    # random_image2 = random_image2.int()


    feat = mymodel(random_image)
    feat2 = mymodel(random_image2)
    print(feat.shape)
    # diff = torch.abs(feat-feat2)
    # print(diff.shape)
    # diff2 = torch.sum(diff, dim=1)
    # reshaped_tensor = torch.reshape(diff2, (1,112*112))
    # print(reshaped_tensor.shape)
    def mse(x,y):
        # Compute the squared error
        # Compute the mean over all pixels and all images in the batches
        batchzise =  max(x.size(0),y.size(0))
        # or here I can do the inverse DCT
        diff = torch.abs(x - y)
        distance_images  = _dct_to_images(diff)
        mse_per_image_pair = torch.sum(distance_images.view(batchzise, -1), dim=1)
        return mse_per_image_pair
    squared_error = mse(feat, feat2)
    print(squared_error)
    print(squared_error.shape,'***')

