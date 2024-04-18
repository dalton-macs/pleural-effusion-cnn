"""
Put each architecture code here
"""
from typing import List
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
import torch


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
                     'Conv2d_1a_3x3',
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

        # Modify the convolution layer to take in grayscale image (1 channel)
        self.inception_v3.Conv2d_1a_3x3.conv = nn.Conv2d(1,
                                                         32,
                                                         kernel_size=(3, 3),
                                                         stride=(2, 2),
                                                         bias=False)

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


# TODO: Jingni
# define Dense Layer
class _DenseLayer(nn.Module):
    def __init__(
            self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()

        self.norm1 = nn.BatchNorm2d(num_features=num_input_features)
        # num_parameters=1/num_of_channels, init=a - learnable parameter
        self.relu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.conv1 = nn.Conv2d(in_channels=num_input_features,
                               out_channels=bn_size * growth_rate,
                               kernel_size=1,
                               stride=1,
                               bias=False)

        self.norm2 = nn.BatchNorm2d(num_features=bn_size * growth_rate)
        self.relu2 = nn.PReLU(num_parameters=1, init=0.25)
        self.conv2 = nn.Conv2d(in_channels=bn_size * growth_rate,
                               out_channels=growth_rate,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        return new_features


# define Dense Block which consist of multiple Dense Layers
class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


# define Transition Layer - 1
# the layer between the dense blocks: consist of the conv layer and pooling
# DenseNet-121: batch-norm -> 1x1 conv -> 2x2 avg pooling
class _Transition1(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


# define Transition Layer - 2
# the layer between the dense blocks: consist of the conv layer and pooling
class _Transition2(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


# define SE block
# Squeeze and Excitation Class
class _SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(_SEBlock, self).__init__()
        # global avg pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        # FC-MLP
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLu(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid(),
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.mlp(y).vide(b, c, 1, 1)
        return x * y.expand_as(x)


# define DenseNet architecture
class DenseNetCustom(nn.Module):
    """
    Args:
        growth_rate (int): how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) : how many layers in each pooling block. DenseNet 121: (6, 12, 24, 16)
        num_init_features (int): the number of filters to learn in the first convolution layer
        bn_size (int): multiplicative factor for number of bottle neck layers
                      (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float): dropout rate after each dense layer
        num_classes (int): number of classification classes
        memory_efficient (bool): If True, uses checkpointing. Much more memory efficient, but slower. Default: False.
    """
    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=2,
                 memory_efficient=False):

        super(DenseNetCustom, self).__init__()

        # First Convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        # Add multiple dense blocks based on config
        # for densenet-121 config: [6,12,24,16]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # add transition layer between dense blocks to
                # down sample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
