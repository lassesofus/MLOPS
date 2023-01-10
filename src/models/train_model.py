import sys
sys.path.append("/mnt/c/Users/Lasse/Desktop/DTU/7. semester/MLOps/MLOPS")
import argparse
import torch
import click
import torch
from src.data.dataset import MyDataset
from model import MyAwesomeModel
from src.visualization.utils import plot_loss
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm



@click.group()
def cli():
    """
    This function creates a command line interface (CLI) group, which can be used to add multiple commands.
    """
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    """
    This function is the main training function for the model.
    It takes a single argument, 'lr', which represents the learning rate for training.
    This argument is optional and defaults to 1e-3 if not provided.
    """
    print("Training day and night")
    print(lr)

    # Defining device on which the training will take place.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # Initializing the model and defining the criterion and optimizer
    model = MyAwesomeModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_set = MyDataset("train")
    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)

    epochs = 10

    train_losses = []
    # Set the model in training mode
    model.train()
    # Loop over the dataset multiple times
    for e in tqdm(range(epochs)):
        running_loss = 0

        # Loop over the batches in the train_dl
        for images, labels in train_dl:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Training loss in epoch {e+1}: {running_loss}")
        train_losses.append(running_loss)

    # Using the visualization utility to plot the loss
    plot_loss(list(range(epochs)), list(train_losses))
    # Saving the trained model
    torch.save(model.state_dict(), "/mnt/c/Users/Lasse/Desktop/DTU/7. semester/MLOps/MLOPS/models/checkpoint.pth")

        

cli.add_command(train)


if __name__ == "__main__":

    cli()

