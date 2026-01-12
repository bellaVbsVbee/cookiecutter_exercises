from s2m6_ccex.model import Model
from s2m6_ccex.data import corrupt_mnist
# from torch import nn
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import typer
import os

import hydra
import logging


root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


log = logging.getLogger(__name__)

@hydra.main(config_path = os.path.join(os.path.dirname(__file__), "conf"), config_name = "config")
def train(cfg):

    torch.manual_seed(cfg.training.hyperparameters.seed)

    trainset, _ = corrupt_mnist()
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=cfg.training.hyperparameters.batch_size, 
                                              shuffle=True)

    model = Model(cfg)
    # add rest of your training code here

    criterion = nn.NLLLoss()
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())

    train_losses = [] # Loss per step
    train_steps = []
    step = 1
    
    for i in range(cfg.training.hyperparameters.epochs):
        print(f"Epoch {i}")
        for images, labels in trainloader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_steps.append(step)
            step += 1
    
    # torch.save(model.state_dict(), "~/exercises/cookiecutter_exercises/models/corrupt_mnist_checkpoint.pth")
    models_dir = os.path.join(root, "models")
    save_path = os.path.join(models_dir, "corrupt_mnist_checkpoint.pth")
    torch.save(model.state_dict(), save_path)
    # torch.save(model.state_dict(), "models/corrupt_mnist_checkpoint.pth")

    plt.figure()
    plt.plot(train_steps, train_losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training loss per step")
    # plt.savefig("~/exercises/cookiecutter_exercises/reports/figures/conv_2_lin_3_lr_1e4.png")
    reports_dir = os.path.join(root, "reports")
    save_path = os.path.join(reports_dir, "figures/training_loss.png")
    plt.savefig(save_path)
    # plt.savefig("reports/figures/conv_2_lin_3_lr_1e4.png")

if __name__ == "__main__":
    train()
