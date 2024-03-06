import os
import torch
import sys
from termcolor import cprint
from collections import OrderedDict
import torch.nn as nn

from encoder.InsightFace_Pytorch.model import Backbone
from encoder.InsightFace_Pytorch import config
from encoder.MagFace.models import iresnet
from encoder.AdaFace import net as AdaFacenet
from encoder.DuetFace.DCTModel import DCTModel

adaface_models = {
    'ir_50': "/media/Storage2/zh/face-privacy/TFace-master/recognition/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
    # 'ir_101':"pretrained/adaface_ir101_ms1mv3.ckpt",
}

# adaface
def load_adaface(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = AdaFacenet.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location='cpu')['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict, False)
    model.eval()
    return model

# arcface
def load_arcface(config):
    # 'num_layers should be 50,100, or 152'
    # model = Backbone(50, drop_ratio=0.1, mode='ir')
    model = Backbone(config.get_config().net_depth, config.get_config().drop_ratio, config.get_config().net_mode)
    model.load_state_dict(torch.load("/media/Storage2/zh/face-privacy/TFace-master/recognition/InsightFace_Pytorch/model_ir_se50.pth", map_location='cpu'), False)
    return model

#magface
def load_features(args):
    if args['arch'] == 'iresnet34':
        features = iresnet.iresnet34(
            pretrained=False,
            num_classes=args['embedding_size'],
        )
    elif args['arch'] == 'iresnet18':
        features = iresnet.iresnet18(
            pretrained=False,
            num_classes=args['embedding_size'],
        )
    elif args['arch'] == 'iresnet50':
        features = iresnet.iresnet50(
            pretrained=False,
            num_classes=args['embedding_size'],
        )
    elif args['arch'] == 'iresnet100':
        features = iresnet.iresnet100(
            pretrained=False,
            num_classes=args['embedding_size'],
        )
    else:
        raise ValueError()
    return features
def load_dict_inf(args, model):
    if os.path.isfile(args['resume']):
        cprint('=> loading pth from {} ...'.format(args['resume']))
        # if args.cpu_mode:
        #     checkpoint = torch.load(args.resume, map_location='cpu')
        # else:
        checkpoint = torch.load(args['resume'], map_location='cpu')
        _state_dict = clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(args['resume']))
    return model
def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        # # assert k[0:1] == 'features.module.'
        new_k = 'features.'+'.'.join(k.split('.')[2:])
        if new_k in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        # assert k[0:1] == 'module.features.'
        new_kk = '.'.join(k.split('.')[1:])
        if new_kk in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
            num_model, num_ckpt))
    return _state_dict


class NetworkBuilder_inf(nn.Module):
    def __init__(self, args):
        super(NetworkBuilder_inf, self).__init__()
        self.features = load_features(args)

    def forward(self, input):
        # add Fp, a pose feature
        x = self.features(input)
        return x

def load_magface(conf):
    # python -u trainer.py \
    # --arch ${MODEL_ARC} \
    # --train_list /training/face-group/opensource/ms1m-112/ms1m_train.list \
    # --workers 8 \
    # --epochs 25 \
    model = NetworkBuilder_inf(conf)
    # Used to run inference
    model = load_dict_inf(conf, model)
    return model
#magface

def load_dctface(conf):
    # python -u trainer.py \
    # --arch ${MODEL_ARC} \
    # --train_list /training/face-group/opensource/ms1m-112/ms1m_train.list \
    # --workers 8 \
    # --epochs 25 \
    model = DCTModel()
    # Used to run inference
    return model
#magface
