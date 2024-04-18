"""
The main training file
"""
import click
from train_wrappers import (
    ResNet18CustomShowkatWrapper,
    GoogLeNetTangWrapper,
)


# TODO: Make CLI more customizable while preserving defaults

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
        click.echo(f'Loading model checkpoint from {checkpoint_path}')
        model, start_epoch = wrapper.wrap_from_checkpoint(checkpoint_path)

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
        click.echo(f'Loading model checkpoint from {checkpoint_path}')
        model, start_epoch = wrapper.wrap_from_checkpoint(checkpoint_path)

    batch_size = wrapper.batch_size
    n_epochs = wrapper.epochs

    click.echo(f'Training the Custom GoogLeNet Model')
    model.fit(n_epochs=n_epochs, start_epoch=start_epoch,
              batch_size=batch_size)


# TODO: Jeffrey or Jingni
@cli.command(name='unet')
# @click.option('--model-name', default=None, help='model_name')
def train_unet():
    click.echo(f'Training the Custom UNet Model')
    raise NotImplementedError


# TODO: Jeffrey or Jingni
@cli.command(name='densenet')
# @click.option('--model-name', default=None, help='model_name')
def train_densenet():
    click.echo(f'Training the Custom DenseNet Model')
    raise NotImplementedError

if __name__ == '__main__':
    cli()