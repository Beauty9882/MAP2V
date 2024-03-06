import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self, vgg, feature_layer):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(vgg.children())[:feature_layer])
        
    def forward(self, x):
        return self.features(x)

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer = 10):
        super(PerceptualLoss, self).__init__()
        self.extractor = FeatureExtractor(models.vgg16(pretrained=True).features.eval(), feature_layer).eval()
        self.criterion = nn.MSELoss()
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        
    def forward(self, input, target):
        input = self.transform(input)
        target = self.transform(target)

        features_input = self.extractor(input)
        features_target = self.extractor(target)
        loss = self.criterion(features_input, features_target)
        return loss

# # Create an instance of the PerceptualLoss
# perceptual_loss = PerceptualLoss()

# # Load and preprocess the images
# transform = transforms.Compose([transforms.Resize((224, 224)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                      std=[0.229, 0.224, 0.225])])

# image1 = transform(Image.open('image1.jpg')).unsqueeze(0)
# image2 = transform(Image.open('image2.jpg')).unsqueeze(0)

# # Calculate the perceptual loss
# loss = perceptual_loss(image1, image2)

# print(f'Perceptual loss: {loss.item()}')
