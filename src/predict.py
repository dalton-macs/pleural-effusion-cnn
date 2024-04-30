import os
import click
from datetime import datetime
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
import torch

from dotenv import load_dotenv
from src.wrappers import (
    ResNet18CustomShowkatWrapper,
    GoogLeNetTangWrapper,
    DenseNetWrapper,
    UNetWrapper,
    BaselineWrapper
)
from src.cnn_pe import BaseCNNPE
from utils import load_model_from_checkpoint

load_dotenv()
AWS_BUCKET = os.getenv('AWS_BUCKET')
USE_AWS =os.getenv('USE_AWS', 'False').lower() == 'true'
LOCAL_DATA_PATH = os.getenv('LOCAL_DATA_PATH')
MODEL_PREFIX = os.getenv('MODEL_PREFIX')
HF_DATASET_TRAIN_SIZE = int(os.getenv('HF_DATASET_TRAIN_SIZE'))
HF_DATASET_VALID_SIZE = int(os.getenv('HF_DATASET_VALID_SIZE'))
HF_DATASET_TEST_SIZE = int(os.getenv('HF_DATASET_TEST_SIZE'))
DATASET = load_dataset(os.getenv('HF_DATASET'), streaming=True)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@click.group()
def cli():
    pass

@cli.command(name='predict')
@click.option('--model-name',
              default=None,
              type=str,
              help='name of model')
@click.option('--model-path',
              default=None,
              type=str,
              help='Path to model checkpoint')
@click.option('--output-path',
              default='./data/models',
              type=str,
              help='Path to save predictions')
def predict(model_name, model_path, output_path = './data/models',
            save_output = True, dataset = None):
    match model_name.lower():
        case 'resnet':
            wrapper = ResNet18CustomShowkatWrapper()
        case 'googlenet':
            wrapper = GoogLeNetTangWrapper()
        # TODO: Implement these wrappers and put here
        case 'unet':
            wrapper = UNetWrapper()
        case 'densenet':
            wrapper = DenseNetWrapper()
        case _:
            raise ValueError

    model = load_model_from_checkpoint(model_path, DEVICE)
    dataset = DATASET['test'] if dataset is None else dataset
    transform = wrapper.transform
    batch_size = wrapper.batch_size

    labels, preds = BaseCNNPE.predict(model, dataset, transform, batch_size)

    df = pd.DataFrame({'labels': labels, 'predictions': preds})

    if save_output:
        current_time = datetime.now().strftime('%Y%m%d%H%M')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        df.to_csv(os.path.join(output_path, 
                               f"{model_name}_test_predictions_{current_time}.csv"))
        
    return df

@cli.command(name='predict-baseline')
@click.option('--model-name',
              default=None,
              type=str,
              help='name of model')
@click.option('--model-path',
              default=None,
              type=str,
              help='Path to model checkpoint')
@click.option('--output-path',
              default='./data/models',
              type=str,
              help='Path to save predictions')
def predict_baseline(model_name, model_path, output_path = './data/models',
            save_output = True, dataset = None):

    from src.architectures import SimpleCNN


    model = SimpleCNN(num_classes=2).to(DEVICE)
    wrapper = BaselineWrapper()
    checkpoint = load_model_from_checkpoint(model_path, DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    dataset = DATASET['test'] if dataset is None else dataset
    transform = wrapper.transform
    batch_size = wrapper.batch_size

    labels, preds = BaseCNNPE.predict(model, dataset, transform, batch_size)

    df = pd.DataFrame({'labels': labels, 'predictions': preds})

    if save_output:
        current_time = datetime.now().strftime('%Y%m%d%H%M')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        df.to_csv(os.path.join(output_path, 
                               f"{model_name}_test_predictions_{current_time}.csv"))
        
    return df

if __name__ == '__main__':
    cli()

    # python src/predict.py predict --model-name googlenet --model-path /mnt/d/mimic-cxr-jpg/model/GoogLeNetTangCustom-1GPU-dmacres/GoogLeNetTangCustom-1GPU-dmacres-fully-trained.pth
    # python src/predict.py predict --model-name resnet --model-path /mnt/d/mimic-cxr-jpg/model/ResNet18CustomShowkatWrapper-1GPU-dmacres/ResNet18CustomShowkatWrapper-1GPU-dmacres-fully-trained.pth
    # python src/predict.py predict --model-name unet --model-path /mnt/d/mimic-cxr-jpg/model/UNet/UNetCustom-Attention-fully-trained.pth
    # python src/predict.py predict --model-name densenet --model-path /mnt/d/mimic-cxr-jpg/model/DenseNet/DenseNetCustom-fully-trained.pth
    # python src/predict.py predict-baseline --model-name baseline --model-path /mnt/d/mimic-cxr-jpg/model/simple-cnn-poc/simple-cnn-poc-fully-trained.pth
