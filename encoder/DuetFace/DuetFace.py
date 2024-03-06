import torch
import sys
import torch.nn as nn

from scipy.spatial import ConvexHull, Delaunay
import torch
import torch.nn.functional as F
from torchjpeg import dct
import numpy as np

from encoder.DuetFace.local_backbones import ClientBackbone, ServerBackbone
from typing import Dict, Iterable, Callable


class InteractiveBlock(nn.Module):
    def __init__(self,device):
        super(InteractiveBlock, self).__init__()
        self.activation = nn.Sigmoid()
        self.interface = None
        self.weight = nn.Parameter(torch.tensor(0.))
        self.device = device
        # self.interface = nn.Sequential(
        #     nn.Upsample(size=(56, 56), mode='bilinear'),
        #     nn.Conv2d(64, 64, (1, 1))
        # ) #.to(self.device)

    def forward(self, x):

        def reshape_and_normalize_masks(inputs, to_bool=False, squeeze=True):
            if squeeze:
                mask = inputs.mean(1)
                mask = mask.unsqueeze(dim=1)
            else:
                mask = inputs
            n, _, h, w = mask.size()
            batch_min, batch_max = [], []
            for i in range(n):
                min, max = mask[i].min().item(), mask[i].max().item()
                img_min, img_max = torch.full((h, w), min), torch.full((h, w), max)
                # img_min, img_max = torch.full((h, w), min).cuda(), torch.full((h, w), max).cuda()
                img_min, img_max = img_min.unsqueeze(dim=0), img_max.unsqueeze(dim=0)
                img_min, img_max = img_min.unsqueeze(dim=0), img_max.unsqueeze(dim=0)  # yes, do it twice (HW -> NCHW)
                batch_min.append(img_min)
                batch_max.append(img_max)

            batch_min = torch.cat(batch_min, dim=0).to(self.device)
            batch_max = torch.cat(batch_max, dim=0).to(self.device)
            mask = (mask - batch_min) / (batch_max - batch_min)
            if to_bool:
                mask = (mask > 0.5).float()  # turn into boolean mask
            return mask

        if len(x) == 3:
            main_inputs, embedding_inputs, inference_x = x[0], x[1], x[2]
        else:
            # deprecated
            main_inputs, embedding_inputs, inference_x = x[0], x[1], None

        shape = main_inputs.shape

        # dynamically produced to meet the client-side shape
        self.interface = nn.Sequential(
            nn.Upsample(size=(shape[2], shape[3]), mode='bilinear'),
            nn.Conv2d(embedding_inputs.shape[1], shape[1], (1, 1))
        ).to(self.device) # very confused! why claim it here???? should be in the init!
        embedding_inputs = self.interface(embedding_inputs)

        # mask can be smaller than 0, so align to [0, 1] first; reuse reshape_and_normalize_masks for simplicity
        mask = reshape_and_normalize_masks(embedding_inputs)

        # crop mask with the convex hull of facial landmarks to acquire ROI
        if inference_x is not None:
            scale_factor = mask.shape[2] / inference_x.shape[2]
            inference_mask = F.interpolate(inference_x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            mask = mask * inference_mask.to(self.device)

        # again, align cropped mask to [0, 1]
        mask = reshape_and_normalize_masks(mask, squeeze=False)

        main_inputs = self.activation(main_inputs)

        # overlay mask to server-side feature maps
        main_outputs = main_inputs * mask * self.weight + main_inputs

        return main_outputs

class DuetFaceBasicBlock(nn.Module):
    """ BasicBlock for IRNet
    """

    def __init__(self, in_channel, depth, stride, feature_channel, kernel_size, stage=0, embedding=False, device = None):
        super(DuetFaceBasicBlock, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))
        if embedding:
            self.embedding_layer = InteractiveBlock(device).to(device)
        else:
            self.embedding_layer = None
        self.stage = stage

    def forward(self, x):
        if len(x) == 2:
            main_x, embedding_x = x[0], x[1]

            shortcut = self.shortcut_layer(main_x)
            res = self.res_layer(main_x)
            main_x = shortcut + res

            if self.embedding_layer is not None:
                main_x = self.embedding_layer([main_x, embedding_x[self.stage]])
            return [main_x, embedding_x]
        else:  # len(x) == 3
            main_x, embedding_x, inference_x = x[0], x[1], x[2]

            shortcut = self.shortcut_layer(main_x)
            res = self.res_layer(main_x)
            main_x = shortcut + res
            if self.embedding_layer is not None:
                main_x = self.embedding_layer([main_x, embedding_x[self.stage], inference_x])
            return [main_x, embedding_x, inference_x]



class DuetFaceModel(nn.Module):
    def __init__(self, num_sub_channels, len_features, len_sub_features, landmark_inference=None, main_model_name='IR_18',
                 sub_model_name='MobileFaceNet',sub_channels = [ 0, 1, 2, 3, 4, 5, 8, 9, 16, 24 ] ,device =None):
        super(DuetFaceModel, self).__init__()
        if main_model_name == 'IR_18':
            model_size = 18
        else:
            model_size = 50
        # reshape in and output, feature length, and override the server-side unit module
        self.server_model = ServerBackbone([112, 112], model_size, 192 - num_sub_channels, len_features,
                                           len_sub_features, kernel_size=3, device =device, unit_module=DuetFaceBasicBlock)
        self.client_model = ClientBackbone(channels_in=num_sub_channels, channels_out=len_sub_features)
        self.landmark_model = landmark_inference
        self.sub_channels = sub_channels
        

    def get_activation(self,name) -> Callable:
        def hook(model, input, output):
            # detach, otherwise the activation will not be removed during backward
            self._activation[name] = output.detach()
        return hook
    
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

    
    def forward(self, x , warm_up=False):
        x_server, x_client = self._images_to_dct(x, sub_channels=self.sub_channels)#main_inputs:[30,162,112,112] sub_inputs:[30,30,112,112]
        if self.landmark_model is not None:
            landmark_inference = self.calculate_landmarks(x)#[30,1,112,112]
        else:
            landmark_inference = None

        if warm_up:
            sub_features = self.client_model(x_client)
            return sub_features
        else:
            # freeze the sub-model after pretraining
            for param in self.client_model.parameters():
                param.requires_grad = False

            self._activation = {}


                    # hook the feature maps of intermediate layers to retrieve attention
            body_blocks = list(self.client_model.client_backbone.children())[1:-1]  # retain only bottleneck blocks
            self.handles = {}
            for i in range(len(body_blocks)):
                name = 'body_{}'.format(str(i))
                self.handles[name] = body_blocks[i].register_forward_hook(hook = self.get_activation(name))

            # perform forward to obtain activation for feature masks
            _ = self.client_model(x_client) # as it is useless for ours work, we ignore this op. !!!NOOO, we can not, it needs to be hooked

            intermediate_output = []
            for i in [1, 3, 5, 7]:
                output = self._activation['body_{}'.format(str(i))]# activation -> handles
                intermediate_output.append(output)
            if landmark_inference is not None:
                main_features = self.server_model([x_server, intermediate_output, landmark_inference])
            else:
                main_features = self.server_model([x_server, intermediate_output])

            # clear activation and remove hook
            self._activation.clear()# may be the reason!
            for key, _ in self.handles.items():
                self.handles[key].remove()

            return main_features
    def calculate_landmarks(self, inputs):
        size = 112
        inputs = inputs * 0.5 + 0.5  # PFLD requires inputs to be within [0, 1]
        _, landmarks = self.landmark_model(inputs)
        # landmarks = landmarks.detach().cpu().numpy()
        # print(landmarks,landmarks.shape)
        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
        landmarks = landmarks * size
        landmark_masks = torch.zeros((landmarks.shape[0], size, size),device=landmarks.device)

        def in_hull(p, hull):
            #  test if points in `p` are in `hull`
            if not isinstance(hull, Delaunay):
                # hull = Delaunay(hull.cpu())
                hull = Delaunay(hull.detach().cpu().numpy())
            return hull.find_simplex(p) >= 0

        x, y = np.mgrid[0:size:1, 0:size:1]
        grid = np.vstack((y.flatten(), x.flatten())).T  # swap axes
        for i in range(len(landmarks)):
            # hull = ConvexHull(landmarks[i].cpu())
            hull = ConvexHull(landmarks[i].detach().cpu().numpy())
            points = landmarks[i, hull.vertices, :]
            mask = torch.from_numpy(in_hull(grid, points).astype(int).reshape(size, size)).unsqueeze(0)
            landmark_masks[i] = mask
        landmark_masks = landmark_masks.unsqueeze(dim=1)
        landmark_masks.requires_grad = False

        return landmark_masks