# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import StandardScaler
import os
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_files, test_files = [], []

    for root, dirs, file in os.walk(input_filepath):
        if file[:4] == "train":
            train_files.append(np.load(file))
        elif file[:4] == "test":
            test_files.append(np.load(file))
            

    data_path = "C:/Users/Lasse/Desktop/MLOPS/dtu_mlops/data/corruptmnist/"
    train_paths = [data_path+"train_"+str(i)+".npz" for i in range(5)]
    test_path = data_path+"test.npz"

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



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
