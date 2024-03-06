# based on:
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/backbone/model_resnet.py
import os
import sys
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
import torch.nn.functional as F
from torchjpeg import dct
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import ReLU
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Module
# from recognition.torchkit.backbone.common import initialize_weights
from torchkit.backbone.common import initialize_weights


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding
    """
    return Conv2d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution
    """
    return Conv2d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=False)


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(Module):
    """ ResNet backbone
    """
    def __init__(self, input_size, block, layers, zero_init_residual=True):
        """ Args:
            input_size: input_size of backbone
            block: block function
            layers: layers in each block
        """
        super(ResNet, self).__init__()
        assert input_size[0] in [112, 224],\
            "input_size should be [112, 112] or [224, 224]"
        self.inplanes = 64
        self.conv1 = Conv2d(189, 64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn_o1 = BatchNorm2d(2048)
        self.dropout = Dropout()
        if input_size[0] == 112:
            self.fc = Linear(2048 * 4 * 4, 512)
        else:
            self.fc = Linear(2048 * 7 * 7, 512)
        self.bn_o2 = BatchNorm1d(512)


        initialize_weights(self.modules())
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)
    
    def _images_to_dct(self, x):
        x = (x + 1) / 2 * 255
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        if x.shape[1] != 3:
            print("Wrong input, Channel should equals to 3")
            return
        x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
        x -= 128
        bs, ch, h, w = x.shape
        block_num = h // 8
        x = x.view(bs * ch, 1, h, w)
        x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
                    stride=(8, 8))
        x = x.transpose(1, 2)
        x = x.view(bs, ch, -1, 8, 8)
        dct_block = dct.block_dct(x)
        dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
        dct_block = dct_block[:, :, 1:, :, :]  # remove DC
        dct_block = dct_block.reshape(bs, -1, block_num, block_num)
        return dct_block
    
    def forward(self, x):
        x = self._images_to_dct(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_o1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_o2(x)

        return x


def ResNet_50(input_size, **kwargs):
    """ Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ResNet_101(input_size, **kwargs):
    """ Constructs a ResNet-101 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def ResNet_152(input_size, **kwargs):
    """ Constructs a ResNet-152 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 8, 36, 3], **kwargs)

    return model
