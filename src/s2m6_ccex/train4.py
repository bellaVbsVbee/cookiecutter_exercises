# Testing the lightning module

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from s2m6_ccex.model3 import Model
from s2m6_ccex.data import corrupt_mnist
import torch
import os
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split


import wandb


root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def train():


    wandb_logger = WandbLogger(log_model="all")
    cfg = wandb_logger.experiment.config
    
    torch.manual_seed(cfg.seed)

    trainset, testset = corrupt_mnist()

    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_ds, val_ds = random_split(trainset, [train_size, val_size])


    trainloader = DataLoader(train_ds, 
                            batch_size=cfg.batch_size, 
                            shuffle=True)
    valloader = DataLoader(val_ds, 
                            batch_size=cfg.batch_size)
    testloader = DataLoader(testset, 
                            batch_size=cfg.batch_size)

    model = Model()
    
    models_dir = os.path.join(root, "models")
    checkpoint_callback = ModelCheckpoint(
                            dirpath=models_dir, filename= "corrupt_mnist_checkpoint.pth", monitor="train_loss", mode="min"
                        )
    early_stopping_callback = EarlyStopping(
                            monitor="train_loss", patience=3, verbose=True, mode="min"
                        )
    
    trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback], max_epochs = 10,
                        limit_train_batches=0.5, logger=wandb_logger, accelerator="auto")
    
    trainer.fit(model, trainloader, valloader)
    trainer.test(model, testloader)


if __name__ == "__main__":
    train()
