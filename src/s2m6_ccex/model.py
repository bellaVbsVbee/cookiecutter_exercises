from torch import nn
from torch import log_softmax
import torch

class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        # self.layer = nn.Linear(1, 1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.pool(self.conv1(x))
        h2 = self.pool(self.conv2(h1))
        h3 = torch.flatten(h2, start_dim=1)
        h4 = self.dropout(self.relu(self.fc1(h3)))
        h5 = self.relu(self.fc2(h4))
        # return self.layer(x)
        return log_softmax(self.fc3(h5), dim=1)

if __name__ == "__main__":
    model = Model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

