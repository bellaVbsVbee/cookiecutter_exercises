from torch import nn, optim
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
import torch
import wandb


class Model(LightningModule):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
                            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = 2, stride = 2),
                            nn.Dropout(p=0.2),
                            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = 2, stride = 2),
                            nn.Dropout(p=0.2),
                            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = 2, stride = 2),
                            nn.Dropout(p=0.2)
                        ) 
        
        self.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.LazyLinear(10),
                            nn.LogSoftmax(dim=1)
                        )
        
        self.criterium = nn.NLLLoss()

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = accuracy_score(target, preds.argmax(dim=1))
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(lr = 1e-3, params=self.parameters())
    
    def test_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
