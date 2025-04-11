import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

import wandb
import yaml
import argparse

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from scripts import BridgeEncoder, get_data

# To put everything on GPU
if(torch.cuda.is_available()):
    use_device = 'cuda:'+str(torch.cuda.current_device())
else:
    use_device = 'cpu'

'''
##################################################################################################
#                                         Parser                                                 #
##################################################################################################
'''

def parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="")
    parser.add_argument("--session", type=Path, default="default", help="Session directory")

    parser.add_argument("--logger", type=str, default="tensorboard", help="Logger.", choices=["tensorboard", "wandb"])

    parser.add_argument("--nodes", type=int, default=1)

    parser.add_argument("--lr", help="Learning rate", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batchsize", type=int, default=64)
    parser.add_argument("--valid_batchsize", type=int, default=32)
    parser.add_argument("--test_batchsize", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser

'''
##################################################################################################
#                                          Main                                                  #
##################################################################################################
'''

def main():
    '''
    Compare with: https://www.nature.com/articles/s41467-023-36736-1
    '''
    # Parse arguments
    args = parser().parse_args()

    # Setting the seed
    pl.seed_everything(args.seed)

    # Input config file
    with open('config.yaml', 'r') as file:
        yaml_input = yaml.safe_load(file)
    
    if(args.logger == 'wandb'):
        wandb.init(entity=yaml_input['wandb_entity'], project=yaml_input['project name'])
        logger = WandbLogger(project=yaml_input['project name'])
    else:
        logger = pl.loggers.TensorBoardLogger('./tensorboard')

    # Create checkpoints
    checkpoints = args.session / 'checkpoints'
    checkpoints.mkdir(exist_ok=True, parents=True)

    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoints,
            filename="BRIDGE_{epoch:04d}-{val_loss:0.3f}-{val_f1:0.3f}",
            monitor="val_loss",
            save_top_k=5,
            save_last=True,
            verbose=True)

    # Trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=torch.cuda.device_count(),
                         num_nodes=args.nodes,
                         logger=logger,
                         max_epochs=args.epochs,
                         gradient_clip_val=5,
                         gradient_clip_algorithm="norm",
                         callbacks=[ LearningRateMonitor("epoch"), checkpoint ],
                         enable_progress_bar=False,
                         #overfit_batches = 1
                         profiler="simple"
                         )

    # Provide learning rate
    model = BridgeEncoder(lr=args.lr)
    model.to(use_device)
    
    # Enable while creating different version of data. such as chaning cutoff etc.
    train, valid = get_data(args.train_batchsize, args.valid_batchsize)

    # Start the training
    trainer.fit(model, train_dataloaders=train, val_dataloaders=valid)
    return True

if(__name__=='__main__'):
    main()