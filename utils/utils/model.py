
from io import BytesIO
import os
import boto3
from dotenv import load_dotenv
import torch

load_dotenv()

AWS_BUCKET = os.getenv('AWS_BUCKET')
AWS_MODEL_PREFIX = os.getenv('AWS_MODEL_PREFIX')

s3 = boto3.client('s3')

def save_model(data, s3_path):
    with BytesIO() as bytes:
        torch.save(data, bytes)
        bytes.seek(0)
        s3.put_object(Body=bytes,
                      Bucket=AWS_BUCKET,
                      Key=f"{AWS_MODEL_PREFIX}/{s3_path}")