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
import pytest
# import wandb

# wandb.login()


root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5, seed: int = 0):

    torch.manual_seed(seed)

    print(f"{lr=}, {batch_size=}, {epochs=}")
    # run = wandb.init(
    #     project="MLOps_corruptmnist",
    #     config={"lr": lr, "batch_size": batch_size, "epochs": epochs, "seed": seed},
    #       )


    trainset, _ = corrupt_mnist()
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size, 
                                              shuffle=True)

    model = Model()
    # add rest of your training code here

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(lr = lr, params=model.parameters())

    train_losses = [] # Loss per step
    train_steps = []
    step = 1
    
    for i in range(epochs):
        print(f"Epoch {i}")
        for images, labels in trainloader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)

            loss.backward()
            optimizer.step()

            accuracy = (log_ps.argmax(dim=1) == labels).float().mean().item()
            # wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            train_losses.append(loss.item())
            train_steps.append(step)
            step += 1
        
        imgs=images[:5].detach().cpu()
        # wandb.log({
        #     "images": [wandb.Image(img) for img in imgs]
        # })

    # final_accuracy = accuracy_score(labels, log_ps.argmax(dim=1))
    # final_precision = precision_score(labels, log_ps.argmax(dim=1), average="weighted")
    # final_recall = recall_score(labels, log_ps.argmax(dim=1), average="weighted")
    # final_f1 = f1_score(labels, log_ps.argmax(dim=1), average="weighted")
    
    # torch.save(model.state_dict(), "~/exercises/cookiecutter_exercises/models/corrupt_mnist_checkpoint.pth")
    models_dir = os.path.join(root, "models")
    save_path = os.path.join(models_dir, "corrupt_mnist_checkpoint.pth")
    torch.save(model.state_dict(), save_path)
    # torch.save(model.state_dict(), "models/corrupt_mnist_checkpoint.pth")
    # artifact = wandb.Artifact(
    #     name="corrupt_mnist_model",
    #     type="model",
    #     description="A model trained to classify corrupt MNIST images",
    #     metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    # )
    # artifact.add_file("model.pth")
    # run.log_artifact(artifact)


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
    return model

if __name__ == "__main__":
    typer.run(train)