from torch import nn
from torch import log_softmax
import torch


class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.fc1 = nn.LazyLinear(10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        
        if x.shape[-2:] != (28,28):
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
    
        h1 = self.dropout(self.pool(self.relu(self.conv1(x))))
        h2 = self.dropout(self.pool(self.relu(self.conv2(h1))))
        h3 = self.dropout(self.pool(self.relu(self.conv3(h2))))
        h4 = torch.flatten(h3, start_dim=1)
        return log_softmax(self.fc1(h4), dim=1)


if __name__ == "__main__":

    model = Model()

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    

