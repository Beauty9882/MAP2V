import torch 
from encoder.PartialFace.model_irse_dct import IR_18, IR_34, IR_50, IR_101, IR_152, IR_200
from encoder.PartialFace.backbone.model_resnet import ResNet_50
from encoder.PartialFace.head import get_head
from encoder.PartialFace.util import get_class_split
import torch.nn.init as init

_model_dict = {
    'IR_18': IR_18,
    'IR_34': IR_34,
    'IR_50': IR_50,
    'IR_101': IR_101,
    'IR_152': IR_152,
    'IR_200': IR_200,
    'ResNet_50': ResNet_50,
}

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

def PartialFaceModel():
    embedding_size = 512
    class_num = 1
    world_size = 1
    rank = 0
    scale = 64
    margin = 0.4
    shape = [112, 112]
    backbone_model = get_model('IR_50')
    backbone = backbone_model(shape)

    # class_shards = []

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
    return backbone

    