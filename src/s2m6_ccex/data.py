from pathlib import Path

import typer
import glob
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")

    train_image_files = glob.glob(os.path.join(data_path, "/train_images*.pt"))
    train_label_files = glob.glob(os.path.join(data_path, "/train_target*.pt"))

    train_images = torch.cat([torch.load(f) for f in train_image_files], dim=0)
    train_labels = torch.cat([torch.load(f) for f in train_label_files], dim=0)

    test_images = torch.load(data_path + "/test_images.pt")
    test_labels = torch.load(data_path + "/test_target.pt")
    
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{output_folder}/corrupt_mnist_v1/train_images.pt")
    torch.save(train_labels, f"{output_folder}/corrupt_mnist_v1/train_target.pt")
    torch.save(test_images, f"{output_folder}/corrupt_mnist_v1/test_images.pt")
    torch.save(test_labels, f"{output_folder}/corrupt_mnist_v1/test_target.pt")

def corrupt_mnist():
    """Return train and test datasets for corrupt MNIST."""
    
    train_images = torch.load("~/data/processed/corrupt_mnist_v1/train_images.pt")
    train_target = torch.load("~/data/processed/corrupt_mnist_v1/train_target.pt")
    test_images = torch.load("~/data/processed/corrupt_mnist_v1/test_images.pt")
    test_target = torch.load("~/data/processed/corrupt_mnist_v1/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
