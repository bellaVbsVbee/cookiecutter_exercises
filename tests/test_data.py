from torch.utils.data import Dataset
from s2m6_ccex.data import corrupt_mnist
import torch
import os.path
import pytest
# from s2m6_ccex.data import MyDataset

@pytest.mark.skipif(not os.path.exists("/home/bella/data/processed/corruptmnist_v1"), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""

    train_images = torch.load("/home/bella/data/processed/corruptmnist_v1/train_images.pt")
    train_target = torch.load("/home/bella/data/processed/corruptmnist_v1/train_target.pt")
    test_images  = torch.load("/home/bella/data/processed/corruptmnist_v1/test_images.pt")
    test_target  = torch.load("/home/bella/data/processed/corruptmnist_v1/test_target.pt")

    assert len(train_images) == 30000 and len(test_images) == 5000

    assert train_images.shape[-2:] == (28,28), "Image shape is not 28x28"
    assert test_images.shape[-2:] == (28,28), "Image shape is not 28x28"

    unique_train_targets = torch.unique(train_target)
    unique_test_targets = torch.unique(test_target)

    assert unique_train_targets.numel() == 10, "The number of classes differes from 10"
    assert unique_train_targets.min() == 0, "The smallest class number is not 0"
    assert unique_train_targets.max() == 9, "The largest class number is not 9"

    assert unique_test_targets.numel() == 10
    assert unique_test_targets.min() == 0
    assert unique_test_targets.max() == 9


if __name__ == "__main__":
    test_my_dataset()
