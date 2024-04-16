from io import BytesIO
import os
import boto3
from torchvision import transforms
import torch
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
AWS_BUCKET = os.getenv('AWS_BUCKET')
s3 = boto3.client('s3')

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