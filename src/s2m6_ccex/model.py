from torch import nn
from torch import log_softmax
import torch
from types import SimpleNamespace

class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self, cfg):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = cfg.model.hyperparameters.conv1_output, 
                               kernel_size = cfg.model.hyperparameters.kernel_size)
        self.pool = nn.MaxPool2d(kernel_size = cfg.model.hyperparameters.pool_kernel, 
                                 stride = cfg.model.hyperparameters.pool_stride)
        self.conv2 = nn.Conv2d(in_channels = cfg.model.hyperparameters.conv1_output, 
                               out_channels = cfg.model.hyperparameters.conv2_output,
                                kernel_size = cfg.model.hyperparameters.kernel_size, 
                                padding = cfg.model.hyperparameters.padding)
        self.conv3 = nn.Conv2d(in_channels = cfg.model.hyperparameters.conv2_output, 
                               out_channels = cfg.model.hyperparameters.conv3_output, 
                               kernel_size = cfg.model.hyperparameters.kernel_size, 
                               padding = cfg.model.hyperparameters.padding)
        self.fc1 = nn.LazyLinear(cfg.model.hyperparameters.output_channels)
        self.dropout = nn.Dropout(p=cfg.model.hyperparameters.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.dropout(self.pool(self.relu(self.conv1(x))))
        h2 = self.dropout(self.pool(self.relu(self.conv2(h1))))
        h3 = self.dropout(self.pool(self.relu(self.conv3(h2))))
        h4 = torch.flatten(h3, start_dim=1)
        return log_softmax(self.fc1(h4), dim=1)


if __name__ == "__main__":
    cfg = SimpleNamespace(
    hyperparameters=SimpleNamespace(
        conv1_output=16,
        conv2_output=32,
        conv3_output=64,
        kernel_size=3,
        padding=1,
        pool_kernel=2,
        pool_stride=2,
        output_channels=10,
        dropout_rate=0.2,
    )
    )
    model = Model(cfg)

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    

