import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import postprocess

class BlackboxEncoder4Adaface(nn.Module):
    """
    Wrapper class for the target encoder.
    Ensures that the gradient is blocked for the encoder.
    This is to follow the black-box assumption.
    """
    def __init__(self, encoder, img_size=112):
        super(BlackboxEncoder4Adaface, self).__init__()
        self.encoder = encoder
        self.resize = transforms.Resize((img_size, img_size))
        self.flip = transforms.RandomHorizontalFlip(p=1.0)
        self.transform = transforms.Compose([
            transforms.Normalize([0, 0, 0], [0.5, 0.5, 0.5])
        ])
    @ torch.no_grad()
    def forward(self, x, flip=True):
        x = self.transform(x)
        x = self.resize(x)
        v,_ = self.encoder(x)
        if flip:
            x_f = self.flip(x)
            v2, _ = self.encoder(x_f)
            v = v *0.5 + v2 *0.5
        return v

class BlackboxEncoder(nn.Module):
    """
    Wrapper class for the target encoder.
    Ensures that the gradient is blocked for the encoder.
    This is to follow the black-box assumption.
    """
    def __init__(self, encoder, img_size=112):
        super(BlackboxEncoder, self).__init__()
        self.encoder = encoder
        self.resize = transforms.Resize((img_size, img_size))
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

    @ torch.no_grad()
    def forward(self, x, flip=True):
        x = self.resize(x)
        v = self.encoder(x)
        if flip:
            x_f = self.flip(x)
            v = v *0.5 + self.encoder(x_f) *0.5
        return v



class WhiteboxEncoder4Adaface(nn.Module):
    """
    Wrapper class for the compromised encoder.
    Unlike BlackboxEncoder, gradient is not blocked.
    """
    def __init__(self, encoder, img_size=112):
        super(WhiteboxEncoder4Adaface, self).__init__()
        self.encoder = encoder
        self.resize = transforms.Resize((img_size, img_size))
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

    def forward(self, x, flip=True):
        x = self.resize(x)
        v,_ = self.encoder(x)
        if flip:
            x_f = self.flip(x)
            v2, _ = self.encoder(x_f)
            v = v *0.5 + v2 *0.5
        return v

class WhiteboxEncoder(nn.Module):
    """
    Wrapper class for the compromised encoder.
    Unlike BlackboxEncoder, gradient is not blocked.
    """
    def __init__(self, encoder, img_size=112):
        super(WhiteboxEncoder, self).__init__()
        self.encoder = encoder
        self.resize = transforms.Resize((img_size, img_size))
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

    def forward(self, x, flip=True):
        x = self.resize(x)
        v = self.encoder(x)
        if flip:
            x_f = self.flip(x)
            v = 0.5*v + 0.5*self.encoder(x_f)
        return v
    

# class WhiteboxEncoder4Due(nn.Module):##due
#     """
#     Wrapper class for the compromised encoder.
#     Unlike BlackboxEncoder, gradient is not blocked.
#     """
#     def __init__(self, encoder, img_size=112):
#         super(WhiteboxEncoder4Due, self).__init__()
#         self.encoder = encoder
#         self.resize = transforms.Resize((img_size, img_size))
#         self.flip = transforms.RandomHorizontalFlip(p=1.0)

#     def forward(self, x, y, z, flip=True):
#         x = self.resize(x)
#         v = self.encoder(x, y, z,warm_up=False)
#         print(v.shape,'======================')
#         if flip:
#             x_f = self.flip(x)
#             v = 0.5*v + 0.5*self.encoder(x_f, y, z, warm_up=False)
#         return v