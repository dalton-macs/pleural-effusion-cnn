"""
Wrap architectures in the BaseCNNPE class.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from cnn_pe import BaseCNNPE, GoogLeNetCNNPE
from architectures import (
    ResNet18Custom,
    GoogLeNetCustom,
    UNetCustom,
    DenseNetCustom,
)
from utils import EarlyStopping, load_model_from_s3_checkpoint

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseWrapper(ABC):
    optimizer_lr: float
    criterion: nn.Module
    transform: transforms.Compose
    batch_size: int
    epochs: int

    @abstractmethod
    def wrap(self) -> Tuple[BaseCNNPE, int]:
        """
        Wraps a model architecture with hyperparameters.

        RETURNS:
        -------
        wrapped architecture : (BaseCNNPE)
            A model ready to be trained

        starting epoch : (int)
            The epoch to start training on again
        """
        pass

    @abstractmethod
    def wrap_from_checkpoint(self,
                             checkpoint_path: str) -> Tuple[BaseCNNPE, int]:
        """
        Loads a model checkpoint from s3 an wraps the architecture
        with hyperparameters.

        PARAMETERS:
        ----------
        checkpoint_path : (str)
            The path to the checkpoint (no bucket)

        RETURNS:
        -------
        wrapped architecture : (BaseCNNPE)
            A model ready to be trained

        starting epoch : (int)
            The epoch to start training on again
        """
                
        pass

    def _put_model_on_device(self):
        self.model = self.model.to(DEVICE)
        # # nn.DataParallel was causing issues/slowing the train time down
        # if torch.cuda.device_count()>1:
        #     self.model = nn.DataParallel(self.model)

    def _compute_start_epoch(self, checkpoint_path: str) -> int:
        """
        Computes the epoch to start with based on a checkpoint path.

        PARAMETERS:
        ----------
        checkpoint_path : (str)
            The path to the checkpoint (no bucket)

        RETURNS:
        starting epoch : (int)
            The epoch to start training on again
        """

        pattern = r"epoch_(\d+)_batch"
    
        # Search the string using the pattern
        match = re.search(pattern, checkpoint_path)
    
        # Convert the matching group (the digits) to an integer if match
        if match:
            # TODO: This assumes the previous epoch finishes
            return int(match.group(1))+1
        else:
            raise ValueError("The path does not contain the expected pattern.")


class ResNet18CustomShowkatWrapper(BaseWrapper):
    """
    A wrapper class to return a trainable object based on the architecture
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

    optimizer_lr: float = 0.0001
    criterion: nn.Module = nn.BCELoss()
    transform: transforms.Compose = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    batch_size: int = 64
    epochs: int = 77

    def __init__(self,
                 model_name: str = 'ResNet18ShowkatCustom',
                 num_classes: int = 2,
                 ) -> None:
        self.model_name = model_name
        self.model = ResNet18Custom(num_classes=num_classes)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.optimizer_lr)
        
    def wrap(self, **kwargs) -> Tuple[BaseCNNPE, int]:

        self._put_model_on_device()

        wrapped_arch = BaseCNNPE(
            model_name=self.model_name,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            transform=self.transform,
            **kwargs
        )
        
        start_epoch = 1

        return wrapped_arch, start_epoch
    
    def wrap_from_checkpoint(self,
                             checkpoint_path: str,
                             **kwargs) -> Tuple[BaseCNNPE, int]:
        
        checkpoint = load_model_from_s3_checkpoint(checkpoint_path, DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._put_model_on_device()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        wrapped_arch = BaseCNNPE(
            model_name=self.model_name,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            transform=self.transform,
            **kwargs
        )

        start_epoch = self._compute_start_epoch(checkpoint_path)

        return wrapped_arch, start_epoch


class GoogLeNetTangWrapper(BaseWrapper):
    """
    A wrapper class to return a trainable object based on the architecture
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

    optimizer_lr: float = 0.001
    lr_momentum: float = 0.9
    criterion: nn.Module = nn.BCELoss()
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        # Convert to three channels or GoogLeNet breaks
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Resize((342, 342)),
        transforms.CenterCrop((299, 299)),
        transforms.RandomHorizontalFlip(),
    ])
    early_stopper: EarlyStopping = EarlyStopping()
    lr_scheduler_kwargs: dict = {'mode': 'min', 'factor': 0.1, 'patience': 5}
    batch_size: int = 64
    epochs: int = 50

    def __init__(self,
                 model_name: str = 'GoogLeNetTangCustom',
                 num_classes: int = 2,
                 ) -> None:
        
        self.model_name = model_name
        self.model = GoogLeNetCustom(num_classes=num_classes)
        self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=self.optimizer_lr,
                                    momentum=self.lr_momentum)
        
    def wrap(self, **kwargs) -> Tuple[BaseCNNPE, int]:
        self._put_model_on_device()

        wrapped_arch = GoogLeNetCNNPE(
            model_name=self.model_name,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            transform=self.transform,
            early_stopper=self.early_stopper,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
            **kwargs
        )
        
        start_epoch = 1

        return wrapped_arch, start_epoch
    
    def wrap_from_checkpoint(self,
                             checkpoint_path: str,
                             **kwargs) -> Tuple[BaseCNNPE, int]:
        
        checkpoint = load_model_from_s3_checkpoint(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._put_model_on_device()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        wrapped_arch = GoogLeNetCNNPE(
            model_name=self.model_name,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            transform=self.transform,
            early_stopper=self.early_stopper,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
            **kwargs
        )

        start_epoch = self._compute_start_epoch(checkpoint_path)

        return wrapped_arch, start_epoch
    