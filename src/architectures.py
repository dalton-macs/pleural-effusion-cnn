"""
Put each architecture code here
"""
from typing import List
import torch.nn as nn
import torchvision.models as models

# TODO: maybe consider implementing transfer learning capabilities if the
# papers mention it

# TODO: Dalton
class ResNet18Custom(nn.Module):
    """
    A custom ResNet18 implementation following the paper below
    https://doi.org/10.1016/j.chemolab.2022.104534

    """

    def __init__(self,
                 num_classes: int,
                 frozen_layers: List[str] = [
                     'bn1',
                     'relu',
                     'maxpool',
                     'layer1',
                     'layer2',
                     'layer3',
                 ]) -> None:
        
        super(ResNet18Custom, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Freeze all parameters in frozen layers
        for name, param in self.resnet18.named_parameters():
            if any(name.startswith(layer) for layer in frozen_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Modify the convolution layer to take in grayscale image (1 channel)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)
        
        # Modify the FC layer based on paper
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=256, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.Linear(in_features=128, out_features=num_classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet18(x)


# TODO: Dalton
class GoogLeNetCustom(nn.Module):
    """
    A Semi-custom Inception-V3 (GoogLeNet) implementation of this paper:
    https://doi.org/10.1038/s41746-020-0273-z
    """
    def __init__(self,
                 num_classes: int,
                 train_layers: List[str] = [
                     'fc'
                 ]) -> None:
        
        super(GoogLeNetCustom, self).__init__()
        self.inception_v3 = models.inception_v3(pretrained=True)
        
        # Freeze all parameters in frozen layers
        for name, param in self.inception_v3.named_parameters():
            if any(name.startswith(layer) for layer in train_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Modify the FC layer to be dynamic to number of classes
        num_features = self.inception_v3.fc.in_features
        self.inception_v3.fc = nn.Sequential(
            nn.Linear(in_features=num_features,
                      out_features=num_classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.inception_v3(x)
    

# TODO: Jeffrey or Jingni
class UNetCustom(nn.Module):
    pass

# TODO: Jeffrey or Jingni
class DenseNetCustom(nn.Module):
    pass