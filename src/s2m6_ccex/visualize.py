import matplotlib.pyplot as plt
import torch
import typer
from s2m6_ccex.model import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from s2m6_ccex.data import corrupt_mnist
from torch import nn
import numpy as np


def visualize(model_checkpoint: str) -> None:

    # Load model
    model = Model()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    model.eval()

    # Load trainset
    _, testset = corrupt_mnist()

    # # Extract intermediate representation from trained network
    # conv_layers = []  # List to store convolutional layers

    # for module in model.features.children():
    #     if isinstance(module, nn.Conv2d):
    #         conv_layers.append(module)

    # # Create figure to show image and feature representation
    # input_image = testset[0]
    # feature_maps = []  # List to store feature maps
    # layer_names = []  # List to store layer names
    # for layer in conv_layers:
    #     input_image = layer(input_image)
    #     feature_maps.append(input_image)
    #     layer_names.append(str(layer))

    # fig = plt.figure(figsize=(30, 50))
    # ax = fig.add_subplot(1, 3, 1)
    # ax.imshow(input_image)
    # ax.axis("off")
    # ax.set_title("Normalized image", fontsize=30)
    # for i in range(len(feature_maps)):
    #     ax = fig.add_subplot(1, 3, i + 2)
    #     ax.imshow(feature_maps[i])
    #     ax.axis("off")
    #     ax.set_title(layer_names[i].split('(')[0], fontsize=30)
    # plt.save("reports/figures/image_through_layers.png")

    # plt.figure()
    # # Visaulize features in 2D space 
    # for i in range(len(feature_maps)):
    #     x = feature_maps[i].view(feature_maps[i].size(0), -1)
    #     x = x.numpy()
    #     tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    #     x_2d = tsne.fit_transform(x)

    #     ax = fig.add_subplot(1, 2, i + 1)
    #     ax.imshow(x_2d)
    #     ax.axis("off")
    #     ax.set_title(layer_names[i].split('(')[0], fontsize=30)
    
    # plt.save("reports/figures/features.png")
    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(testset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/features.png")

if __name__ == "__main__":
    typer.run(visualize)