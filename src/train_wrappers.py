"""
Wrap architectures in the BaseCNNPE class.
"""
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from base_cnn_pe import BaseCNNPE
from architectures import (
    ResNet18Custom,
    GoogLeNetCustom,
    UNetCustom,
    DenseNetCustom,
)

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
    "The customized ResNet model was trained with BCE as the loss function,
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


# TODO: Implement wrappers for all other architectures
