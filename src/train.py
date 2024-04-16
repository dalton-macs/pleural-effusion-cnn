"""
The main training file
"""
import click
from train_wrappers import ResNet18CustomShowkatWrapper

# TODO: Make CLI more customizable while preserving defaults

@click.group()
def cli():
    pass

@cli.command(name='resnet')
# @click.option('--model-name', default=None, help='model_name')
def train_resnet():
    click.echo(f'Training the Custom ResNet Model')
    model, batch_size, n_epochs = ResNet18CustomShowkatWrapper()
    model.fit(n_epochs=n_epochs, batch_size=batch_size)


# TODO: Dalton
@cli.command(name='googlenet')
# @click.option('--model-name', default=None, help='model_name')
def train_googlenet():
    click.echo(f'Training the Custom GoogLeNet Model')
    raise NotImplementedError

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