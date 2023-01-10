import sys
sys.path.append("/mnt/c/Users/Lasse/Desktop/DTU/7. semester/MLOps/MLOPS")
import argparse
import torch
import click
import torch
from src.data.dataset import MyDataset
from src.models.model import MyAwesomeModel
from src.visualization.utils import plot_loss
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



@click.group()
def cli():
    pass

@click.command()
@click.argument("model_checkpoint")
@click.argument("output_path")
def visualize(model_checkpoint, output_path):
    print("Visualizing") 

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")   

    # Load saved model
    model = MyAwesomeModel().to(device)
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    # Load MNIST training dataset
    train_set = MyDataset("train")
    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)

    model.eval()
    with torch.no_grad():
        features = []
        labels_list = []
        for images, labels in tqdm(train_dl):
            images = images.to(device)
            images = images.to(torch.float32)
            # Only feed through first two layers
            images = model.layer1(images)
            images = model.layer2(images)

            images = images.reshape(images.shape[0], -1)
            #labels = labels

            features.append(images)
            labels_list.append(labels)

    features = torch.cat(features, 0).cpu().numpy()
    labels_list = torch.cat(labels_list, 0).cpu().numpy()

    # Perform dimensionality reduction
    print("Performing dimensionalty reduction using TSNE.")
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(features)

    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels_list, cmap="Spectral")
    plt.savefig(output_path)
    plt.show()


cli.add_command(visualize)


if __name__ == "__main__":

    cli()

