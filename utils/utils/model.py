from io import BytesIO
import os
import logging
import boto3
from torchvision import transforms
import torch
from PIL import Image
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()
AWS_BUCKET = os.getenv('AWS_BUCKET')
s3 = boto3.client('s3')


class EarlyStopping:
    """
    A class to implement erly stopping of training if loss plateaus after
    <patience> epochs.
    """

    def __init__(self,
                 patience: int =5,
                 verbose: bool =False,
                 delta: float =0) -> None:
        
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of 
                      {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



def load_model_from_s3_checkpoint(key: str) -> torch.nn.Module:
    """
    Load a PyTorch model from a checkpoint file stored in an Amazon S3 bucket.

    PARAMETERS:
    ----------
        key (str): Key (path) of the checkpoint file in the bucket.

    RETURNS:
    -------
        model: PyTorch model loaded from the checkpoint.
        checkpoint_info (dict): Information stored in the checkpoint
        (e.g., model state dict, optimizer state dict).
    """

    response = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    checkpoint_bytes = response['Body'].read()

    # Load model from checkpoint bytes using BytesIO
    checkpoint_buffer = BytesIO(checkpoint_bytes)

    # Load model from checkpoint buffer
    model = torch.load(checkpoint_buffer)

    return model