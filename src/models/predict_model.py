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
    """Command-line interface entry point"""
    pass


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_path")
def predict(model_checkpoint, data_path):
    """
    Predict the class for input data.

    Args:
        model_checkpoint (str): path to the trained model checkpoint.
        data_path (str): path to the input data.
    """
    print("Predicting")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    model = MyAwesomeModel().to(device)
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    
    # Get scaler from the training data
    train_dataset = MyDataset("train")
    scaler = train_dataset.get_scaler()
    # Load test data and apply the scaler
    test_dataset = MyDataset("test", scaler, data_path=data_path)
    test_dl = DataLoader(test_dataset, batch_size=16, shuffle=True)

    with torch.no_grad():
        # Put the model in evaluation mode
        model.eval() 

        correct = 0
        total = 0
        predictions = []
        for images, _ in tqdm(test_dl):
            images = images.to(device)
            ps = model(images)
            _, top_class = ps.topk(1, dim=1)
            predictions.append(top_class)

    print(predictions)
    print(len(predictions))
    return predictions        

cli.add_command(predict)


if __name__ == "__main__":

    cli()

