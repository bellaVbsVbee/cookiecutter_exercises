# testing wandb

from s2m6_ccex.model2 import Model
from s2m6_ccex.data import corrupt_mnist
# from torch import nn
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import typer
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
import yaml

# wandb.login()

# with open("configs/sweep.yaml", "r") as f:
#     sweep_config = yaml.safe_load(f)

# sweep_id = wandb.sweep(
#     sweep_config, 
#     project="MLOPs_corruptmnist", 
#     entity="mlops_2026"
# )

root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def train():

    with wandb.init() as run:

        cfg = wandb.config
        
        torch.manual_seed(cfg.seed)

        trainset, _ = corrupt_mnist()
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=cfg.batch_size, 
                                                shuffle=True)

        model = Model()
        # add rest of your training code here

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(lr = cfg.learning_rate, params=model.parameters())

        train_losses = [] # Loss per step
        train_steps = []
        step = 1
        
        for i in range(cfg.epochs):
            print(f"Epoch {i}")
            for images, labels in trainloader:
                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)

                loss.backward()
                optimizer.step()

                accuracy = (log_ps.argmax(dim=1) == labels).float().mean().item()
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
                train_losses.append(loss.item())
                train_steps.append(step)
                step += 1
            
        imgs=images[:5].detach().cpu()
        wandb.log({
            "images": [wandb.Image(img) for img in imgs]
        })

        final_accuracy = accuracy_score(labels, log_ps.argmax(dim=1))
        final_precision = precision_score(labels, log_ps.argmax(dim=1), average="weighted")
        final_recall = recall_score(labels, log_ps.argmax(dim=1), average="weighted")
        final_f1 = f1_score(labels, log_ps.argmax(dim=1), average="weighted")
        
        # torch.save(model.state_dict(), "~/exercises/cookiecutter_exercises/models/corrupt_mnist_checkpoint.pth")
        models_dir = os.path.join(root, "models")
        save_path = os.path.join(models_dir, "corrupt_mnist_checkpoint.pth")
        torch.save(model.state_dict(), save_path)
        # torch.save(model.state_dict(), "models/corrupt_mnist_checkpoint.pth")
        artifact = wandb.Artifact(
            name="corrupt_mnist_model",
            type="model",
            description="A model trained to classify corrupt MNIST images",
            metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
        )
        artifact.add_file(save_path)
        run.log_artifact(artifact)


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
    # typer.run(train)
    train()
    # wandb.agent(sweep_id, function=train, count=1)