from s2m6_ccex.model import Model
from s2m6_ccex.data import corrupt_mnist
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
import typer

def evaluate(model_checkpoint: str, batch_size: int = 32):

    _, testset = corrupt_mnist()
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    model = Model()
    
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:

            log_ps = model(images)
            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)
            correct += (top_class.squeeze() == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100}%")



if __name__ == "__main__":
    typer.run(evaluate)
