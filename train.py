"""
Python script for starting the training of the Local and Global Texture Fields
"""

import torch
import argparse
from mesh2tex import config
import matplotlib
torch.set_default_dtype(torch.float32)
matplotlib.use('Agg')


# Create an argumentparser to load config files from the interface
parser = argparse.ArgumentParser(description='Train a Texture Field.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified '
                         'number of seconds with exit code 2.')
args = parser.parse_args()

# Load configuration files 
cfg = config.load_config(args.config, './configs/default.yaml')

# Set the device
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
exit_after = args.exit_after

# Create the models using the settings in configuration files
models = config.get_models(cfg, device=device)
print('Models created')

# Create an optimizer
optimizers = config.get_optimizers(models, cfg)
print('Optimizer created')

# Create training and validation dataloaders
train_loader = config.get_dataloader('train', cfg)
val_loader = config.get_dataloader('val_eval', cfg)


if cfg['training']['vis_fixviews'] is True:
    vis_loader = config.get_dataloader('val_vis', cfg)
else:
    vis_loader = None
print('Train and validation dataloaders created')

# Create a trainer object for the training loop
trainer = config.get_trainer(models, optimizers, cfg, device=device)
print('Trainer created')

# Train the model inside this function
trainer.train(train_loader, val_loader, vis_loader,
              exit_after=exit_after, n_epochs=None)

