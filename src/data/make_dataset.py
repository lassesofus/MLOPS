# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import StandardScaler


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Simply renaming
    data_path = input_filepath

    train_files, test_files = [], []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file[:5] == "train":
                train_files.append(np.load(os.path.join(root, file)))
            elif file[:4] == "test":
                test_files.append(np.load(os.path.join(root, file)))

    # Extract training images and concatenate these into a [25000, 28, 28] numpy ndarray
    train_images = [f["images"] for f in train_files]
    train_images = np.concatenate(train_images)

    scaler = StandardScaler()

    # This normalization is performed on a reshaped array of size [25000, 784] such that each pixel feature is normalized cf. the feature mean and standard deviation
    train_images = scaler.fit_transform(
        train_images.reshape(
            train_images.shape[0], train_images.shape[1] * train_images.shape[2]
        )
    ).reshape(train_images.shape)

    # Add the channel dimension. The resulting dimensions are (25000, 1, 28, 28)
    train_images = torch.from_numpy(train_images).unsqueeze_(1)

    # Extract training labels and concatenate these into a [25000,] numpy ndarray
    train_labels = [f["labels"] for f in train_files]
    train_labels = np.concatenate(train_labels)
    train_labels = torch.from_numpy(train_labels)

    # Extract test images and concatenate these into a [25000, 28, 28] numpy ndarray
    test_images = test_files[0]["images"]
    test_images = scaler.transform(
        test_images.reshape(
            test_images.shape[0], test_images.shape[1] * test_images.shape[2]
        )
    ).reshape(test_images.shape)
    # Add the channel dimension. The resulting dimensions are (5000, 1, 28, 28)
    test_images = torch.from_numpy(test_images).unsqueeze_(1)
    # Extract test labels and concatenate these into a [25000,] numpy ndarray
    test_labels = test_files[0]["labels"]
    test_labels = torch.from_numpy(test_labels)

    train = {"images": train_images, "labels": train_labels}
    test = {"images": test_images, "labels": test_labels}

    torch.save(train, os.path.join(output_filepath, "train.pt"))
    torch.save(test, os.path.join(output_filepath, "test.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
