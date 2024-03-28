
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


def label2numeric(val, di):
    return di[val]


def numeric2label(val, di):
    raise NotImplementedError

def map_labels(example, label_converter):
    example['label'] = label2numeric(example['label'], label_converter)
    return example


# Need to see if we need this
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
])


def get_image(example):
    with BytesIO(s3.get_object(Bucket=AWS_BUCKET, Key=example['file_path'])\
                 ['Body'].read()) as image_data_io:
        image = Image.open(image_data_io)
        image = transform(image)

    return image, example['label'],example['dicom_id']


def collate_get_image(batch):
    images = []
    labels = []
    # dicom_ids = []
    for example in batch:

        image, label, dicom_id = get_image(example)
        images.append(image)
        labels.append(label)
        # dicom_ids.append(dicom_id)
    # Stack the preprocessed images into a batch tensor
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    # dicom_ids = torch.tensor(dicom_ids)
    return {
        'images': images,
        'labels': labels,
        # 'dicom_ids': dicom_ids
    }
