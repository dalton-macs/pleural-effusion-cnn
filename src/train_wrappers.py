"""
Wrap architectures in the BaseCNNPE class.
"""
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from cnn_pe import BaseCNNPE, GoogLeNetCNNPE, DenseNetCNNPE
from architectures import (
    ResNet18Custom,
    GoogLeNetCustom,
    UNetCustom,
    DenseNetCustom,
)
from utils import EarlyStopping


def ResNet18CustomShowkatWrapper(model_name: str = 'ResNet18ShowkatCustom',
                                 num_classes: int = 2,
                                 **kwargs) -> Tuple[BaseCNNPE, int, int]:
    """
    A wrapper function to return a trainable object based on the architecture
    and hyperparameters in the paper below
    https://doi.org/10.1016/j.chemolab.2022.104534

    PARAMETERS:
    ----------
    model_name : str
        name of the model
    num_classes : int
        number of classes in the task
    kwargs : Any
        additional BaseCNNPE arguments (dataset_kwargs and label_map)

    RETURNS:
    -------
    A wrapped model with all hyperparameters : BaseCNNPE
    batch size : int
    number of epochs : int
    
    PAPER INFO:
    ----------
    "All images were rescaled to a size of 224 ​× ​224...
    The customized ResNet model was trained with BCE as the loss function,
    a learning rate of 0.0001, Adam optimizer, and batch size of 64...
    The model was trained with a holdout factor of 35 and trained for 77
    epochs..."
    """

    model_name = model_name
    model = ResNet18Custom(num_classes = num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    batch_size = 64
    epochs = 77

    # TODO: search through kwargs and only keep dataset_kwargs and and label_map
    wrapped_arch = BaseCNNPE(
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        transform=transform,
        **kwargs
    )

    return wrapped_arch, batch_size, epochs


def GoogLeNetTangWrapper(model_name: str = 'GoogLeNetTangCustom',
                                 num_classes: int = 2,
                                 **kwargs) -> Tuple[BaseCNNPE, int, int]:
    """
    A wrapper function to return a trainable object based on the architecture
    and hyperparameters in the paper below
    https://doi.org/10.1038/s41746-020-0273-z

    PARAMETERS:
    ----------
    model_name : str
        name of the model
    num_classes : int
        number of classes in the task
    kwargs : Any
        additional BaseCNNPE arguments (dataset_kwargs and label_map)

    RETURNS:
    -------
    A wrapped model with all hyperparameters : BaseCNNPE
    batch size : int
    number of epochs : int
    
    PAPER INFO:
    ----------
    "For Inception-v3, we resized the image to 342x342 and cropped 299x299 
    enter pixels in order to make it compatible with its original dimensions"

    "using the stochastic gradient descent (SGD) optimizer with the momentum
    of 0.9. The learning rate was reduced by a factor of 0.1 after the loss
    plateaued for five epochs. Early stopping was used to avoid overfitting
    on the training set with a maximum running of 50 epochs.
    The batch size was [64, 128] for an image size of 256x256..."

    "We empirically found for 256x256 input images and a batch size of 64, the
    optimal learning rate was 0.001 for ImageNet pre-trained models...
    We augmented the dataset in the training stage by horizontally flipping 
    the chest radiographs."

    "The loss function was binary cross-entropy loss"
    """

    model_name = model_name
    model = GoogLeNetCustom(num_classes = num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.BCELoss()
    transform = transforms.Compose([
        transforms.Resize((342, 342)),
        transforms.CenterCrop((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    early_stopper = EarlyStopping
    lr_scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5}
    batch_size = 64
    epochs = 50

    # TODO: search through kwargs and only keep dataset_kwargs and and label_map
    wrapped_arch = GoogLeNetCNNPE(
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        transform=transform,
        early_stopper=early_stopper,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        **kwargs
    )

    return wrapped_arch, batch_size, epochs


# TODO: Implement wrappers for all other architectures
def DenseNetWrapper(model_name: str = "DenseNetCustom",
                    num_classes: int = 2,
                    **kwargs):
    model_name = model_name
    model = DenseNetCustom(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    transform = transforms.Compose(
                                    [transforms.Resize((342, 342)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     ]
    )
    early_stopper = EarlyStopping
    lr_scheduler_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5}
    batch_size = 64
    epochs = 10

    wrapped_arch = DenseNetCNNPE(model_name=model_name,
                                 model=model,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 transform=transform,
                                 early_stopper=early_stopper,
                                 lr_scheduler_kwargs=lr_scheduler_kwargs,
                                 **kwargs
    )

    return wrapped_arch, batch_size, epochs