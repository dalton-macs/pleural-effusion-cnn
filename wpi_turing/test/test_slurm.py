import os
import logging
import boto3
from dotenv import load_dotenv
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()
AWS_BUCKET = os.getenv('AWS_BUCKET')
AWS_MODEL_PREFIX = os.getenv('AWS_MODEL_PREFIX')

if __name__ == '__main__':

    logger.info(f"AWS Bucket: {AWS_BUCKET}")
    logger.info(f"AWS Model Prefix: {AWS_MODEL_PREFIX}")

    logger.info(f"Is CUDA available: {torch.cuda.is_available()}")

    s3 = boto3.client('s3')
    response = s3.list_buckets()
    buckets = [bucket['Name'] for bucket in response ['Buckets']]
    buckets_str = ', '.join(buckets)
    logger.info(f"Buckets: {buckets_str}")


