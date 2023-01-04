import argparse
import sys

import torch
import click

from data import MyDataset
from model import MyAwesomeModel
from utils import plot_loss
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_set = MyDataset("train")
    train_dl = DataLoader(train_set, batch_size=16, shuffle=True)

    epochs = 10

    train_losses = []
    # Set the model in training mode
    model.train()
    for e in tqdm(range(epochs)):
        running_loss = 0

        for images, labels in train_dl:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Training loss in epoch {e+1}: {running_loss}")
        train_losses.append(running_loss)
    
    plot_loss(list(range(epochs)), list(train_losses))
    torch.save(model.state_dict(), 'checkpoint.pth')

        
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    
    # Get scaler from the training data
    train_dataset = MyDataset("train")
    scaler = train_dataset.get_scaler()
    test_dataset = MyDataset("test", scaler)
    test_dl = DataLoader(test_dataset, batch_size=16, shuffle=True)

    with torch.no_grad():

        model.eval() 

        correct = 0
        total = 0
        for images, labels in test_dl:
            ps = model(images)
            _, top_class = ps.topk(1, dim=1)
            # Add # images to total for each batch
            total += top_class.shape[0]
            # Determine which predictions are correct
            equals = top_class == labels.view(*top_class.shape)
            correct += equals.sum().item()
        
        accuracy = correct/total
        print(f'Accuracy: {accuracy*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":

    cli()


    
    
    
    