"""
The main training file with click implementation.
"""

import os
from dotenv import load_dotenv
import click
from train_wrappers import (
    ResNet18CustomShowkatWrapper,
    GoogLeNetTangWrapper,
)

load_dotenv()
LOCAL_DATA_PATH = os.getenv('LOCAL_DATA_PATH')


@click.group()
def cli():
    pass

@cli.command(name='resnet')
@click.option('--model-name',
              default=None,
              type=str,
              help='name of model')
@click.option('--checkpoint-path',
              default=None,
              type=str,
              help='S3 path (no bucket) to model checkpoint')
def train_resnet(model_name, checkpoint_path):
    if model_name is not None:
        wrapper = ResNet18CustomShowkatWrapper(model_name=model_name)
    else:
        wrapper = ResNet18CustomShowkatWrapper()

    if checkpoint_path is None:
        model, start_epoch = wrapper.wrap()
    else:
        full_path = os.path.join(LOCAL_DATA_PATH, checkpoint_path)
        click.echo(f'Loading model checkpoint from {full_path}')
        model, start_epoch = wrapper.wrap_from_checkpoint(full_path)


    batch_size = wrapper.batch_size
    n_epochs = wrapper.epochs

    click.echo(f'Training the Custom ResNet Model')
    model.fit(n_epochs=n_epochs, start_epoch=start_epoch,
              batch_size=batch_size)


@cli.command(name='googlenet')
@click.option('--model-name',
              default=None,
              type=str,
              help='name of model')
@click.option('--checkpoint-path',
              default=None,
              type=str,
              help='S3 path (no bucket) to model checkpoint')
def train_googlenet(model_name, checkpoint_path):
    if model_name is not None:
        wrapper = GoogLeNetTangWrapper(model_name=model_name)
    else:
        wrapper = GoogLeNetTangWrapper()

    if checkpoint_path is None:
        model, start_epoch = wrapper.wrap()
    else:
        full_path = os.path.join(LOCAL_DATA_PATH, checkpoint_path)
        click.echo(f'Loading model checkpoint from {full_path}')
        model, start_epoch = wrapper.wrap_from_checkpoint(full_path)

    batch_size = wrapper.batch_size
    n_epochs = wrapper.epochs

    click.echo(f'Training the Custom GoogLeNet Model')
    model.fit(n_epochs=n_epochs, start_epoch=start_epoch,
              batch_size=batch_size)


# TODO: Jeffrey or Jingni
@cli.command(name='unet')
@click.option('--model-name',
              default=None,
              type=str,
              help='name of model')
@click.option('--checkpoint-path',
              default=None,
              type=str,
              help='S3 path (no bucket) to model checkpoint')
def train_unet(model_name, checkpoint_path):
    click.echo(f'Training the Custom UNet Model')
    raise NotImplementedError


# TODO: Jeffrey or Jingni
@cli.command(name='densenet')
@click.option('--model-name',
              default=None,
              type=str,
              help='name of model')
@click.option('--checkpoint-path',
              default=None,
              type=str,
              help='S3 path (no bucket) to model checkpoint')
def train_densenet(model_name, checkpoint_path):
    click.echo(f'Training the Custom DenseNet Model')
    raise NotImplementedError


if __name__ == '__main__':
    cli()
