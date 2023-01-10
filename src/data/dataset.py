import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
import os


mnist_path = "/mnt/c/Users/Lasse/Desktop/DTU/7. semester/MLOps/MLOPS/data/processed/corruptmnist"

# Custom class to instantiate the corrupted mnist dataset
class MyDataset(Dataset):
    def __init__(self, type, scaler=None, data_path=None):
        # Type must be either "train" or "test"
        assert type in ["train", "test"], f"Invalid data type: {type}. Data type must be either 'train' or 'test'."
        if type == "train":
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        # Default dataset to be loaded if no other is provided
        if data_path == None: 
            data_path = mnist_path
        data = torch.load(os.path.join(data_path, str(type + ".pt")))
        self.images = data["images"]
        self.labels = data["labels"]

        # Normalize the data according to the stats of the dataset on which the original model is trained. 
        if type == "test" and self.scaler != None:
            self.images = scaler.transform(self.images)

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return (self.images[index], self.labels[index])
    
    def get_scaler(self):
        return self.scaler




