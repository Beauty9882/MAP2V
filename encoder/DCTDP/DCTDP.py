import torch 
import os
import sys
sys.path.append('/media/Storage2/zh/face-privacy/Black-box-Face-Reconstruction_Hojin/encoder/DCTDP')
from encoder.DCTDP.model_irse_dct import IR_18, IR_34, IR_50, IR_101, IR_152, IR_200
from encoder.DCTDP.backbone.model_resnet import ResNet_50
from encoder.DCTDP.head import get_head
from encoder.DCTDP.util import get_class_split
import torch.nn.init as init
import torch.nn as nn

_model_dict = {
    'IR_18': IR_18,
    'IR_34': IR_34,
    'IR_50': IR_50,
    'IR_101': IR_101,
    'IR_152': IR_152,
    'IR_200': IR_200,
    'ResNet_50': ResNet_50,
}




class NoisyActivation(nn.Module):#用于实现带有噪声的激活函数
    def __init__(self, input_shape=112, budget_mean=2, sensitivity=None):
        super(NoisyActivation, self).__init__()
        self.h, self.w = input_shape, input_shape
        if sensitivity is None:
            sensitivity = torch.ones([189, self.h, self.w])
        self.sensitivity = sensitivity.reshape(189 * self.h * self.w)
        self.given_locs = torch.zeros((189, self.h, self.w))
        size = self.given_locs.shape
        self.budget = budget_mean * 189 * self.h * self.w
        self.locs = nn.Parameter(torch.Tensor(size).copy_(self.given_locs))
        self.rhos = nn.Parameter(torch.zeros(size))
        self.laplace = torch.distributions.laplace.Laplace(0, 1)
        self.rhos.requires_grad = True
        self.locs.requires_grad = True

    def scales(self):
        softmax = nn.Softmax()
        return (self.sensitivity / (softmax(self.rhos.reshape(189 * self.h * self.w))
                * self.budget)).reshape(189, self.h, self.w)

    def sample_noise(self):
        epsilon = self.laplace.sample(self.rhos.shape).cuda()
        return self.locs + self.scales() * epsilon

    def forward(self, input):
        noise = self.sample_noise()
        output = input + noise
        return output

    def aux_loss(self):#根据缩放因子计算辅助损失，该损失是缩放因子的平均值的负对数
        scale = self.scales()
        loss = -1.0 * torch.log(scale.mean())
        return loss


def get_model(key):
    """ Get different backbone network by key,
        support ResNet50, ResNet_101, ResNet_152
        IR_18, IR_34, IR_50, IR_101, IR_152, IR_200,
        IR_SE_50, IR_SE_101, IR_SE_152, IR_SE_200,
        EfficientNetB0, EfficientNetB1.
        MobileFaceNet, FBNets.
    """
    if key in _model_dict.keys():
        return _model_dict[key]
    else:
        raise KeyError("not support model {}".format(key))

def DCTDP_model():
    """ build training backbone and heads
    """

    embedding_size = 512
    class_num = 1
    world_size = 1
    rank = 0
    scale = 64
    margin = 0.4
    shape = [112, 112]
    # backbone_pth = '/media/Storage2/zh/face-privacy/TFace-master/recognition/tasks/dctdp/ckpt/dp/HEAD_Epoch_2_Split_0_checkpoint.pth'
    # state_dict = torch.load(backbone_pth)
    backbone_model = get_model('ResNet_50')
    backbone = backbone_model(shape)
    #backbone = IR_50(shape)
      # set to training mode

    # logging.info("{} Backbone Generated".format(backbone_name))





    class_shards = []

    metric = get_head("ArcFace", dist_fc=True)
    class_shard = get_class_split(class_num, world_size)

    # logging.info('Split FC: {}'.format(class_shard))

    init_value = torch.FloatTensor(embedding_size, class_num)
    init.normal_(init_value, std=0.01)
    head = metric(in_features=embedding_size,
                  gpu_index=rank,
                  weight_init=init_value,
                  class_split=class_shard,
                  scale=scale,
                  margin=margin)
    del init_value


    pth_files = ["/media/Storage2/zh/face-privacy/TFace-master/recognition/ckpt_dp_2/Backbone_Epoch_24_checkpoint.pth", "/media/Storage2/zh/face-privacy/TFace-master/recognition/ckpt_dp_2/HEAD_Epoch_24_Split_0_checkpoint.pth", "/media/Storage2/zh/face-privacy/TFace-master/recognition/ckpt_dp_2/META_Epoch_24_checkpoint.pth"]
    for pth_file in pth_files:
        state_dict = torch.load(pth_file, map_location='cpu')
        backbone.load_state_dict(state_dict, strict=False)

    backbone.eval()
    noise_model = NoisyActivation()
    noise_model.eval()
    head.eval()

    return head, noise_model, backbone

