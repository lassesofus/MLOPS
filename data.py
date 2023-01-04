import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler


data_path = "C:/Users/Lasse/Desktop/MLOPS/dtu_mlops/data/corruptmnist/"
train_paths = [data_path+"train_"+str(i)+".npz" for i in range(5)]
test_path = data_path+"test.npz"


class MyDataset(Dataset):
    def __init__(self, type, scaler=None):
        self.scaler = scaler
        if type == "train":
            # The scaler used for normalization should be based on the training data
            self.scaler = StandardScaler()
            filepaths = train_paths
            images = [np.load(f)["images"] for f in filepaths]
            self.images = np.concatenate(images)
            # Normalize by subtracting mean and dividing with standard deviation across all 784 pixel features
            self.images = self.scaler.fit_transform(self.images.reshape(self.images.shape[0], self.images.shape[1]*self.images.shape[2])).reshape(self.images.shape)
            # Add the channel dimension. The resulting dimensions are (B, C, H, W)
            self.images = torch.from_numpy(self.images).unsqueeze_(1)
            labels = [np.load(f)["labels"] for f in filepaths]
            self.labels = np.concatenate(labels)
        elif type == "test":
            filepaths = test_path
            self.images = np.load(filepaths)["images"]
            self.images = self.scaler.transform(self.images.reshape(self.images.shape[0], self.images.shape[1]*self.images.shape[2])).reshape(self.images.shape)
            self.images = torch.from_numpy(self.images).unsqueeze_(1)
            self.labels = np.load(filepaths)["labels"]
        

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return (self.images[index], self.labels[index])
    
    def get_scaler(self):
        return self.scaler


