import logging
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class BasicDataset(Dataset):
    """
    BasicDataset Class

    Creates basic dataset that returns x and y
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


def get_basic_dataset(x: torch.Tensor, y: torch.Tensor) -> BasicDataset:
    """
    Gets basic dataset

    Parameters:
        x(list): Input data
        y(list): Output data

    Returns:
        basic_dataset(BasicDataset): A basic dataset
    """

    # Get basic dataset
    basic_dataset = BasicDataset(x, y)

    logging.debug("Created BasicDataset")

    return basic_dataset


def get_basic_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """
    Gets basic dataset

    Parameters:
        dataset(Dataset): The input dataset

    Returns:
        dataloader(DataLoader): A basic dataloader
    """

    # Get basic dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    logging.debug("Created DataLoader")

    return dataloader


def get_unprocessed_data(
    data_path: str, train_file_name: str, test_file_name: str
) -> dict:
    """
    Gets unprocessed data as dataframe and returns it in a dictionary.

    Parameters:
        data_path(str): The path to the data folder.
        train_file_name(str): The file name of the train csv file
        test_file_name(str): The file name of the test csv file

    Returns:
        u_train_data(pd.DataFrame): Dictionary containing the unprocessed train dataframes.
        u_test_data(pd.DataFrame): Dictionary containing the unprocessed test dataframes.
    """

    # Read train df
    u_train_data = pd.read_csv(os.path.join(data_path, train_file_name))
    logging.debug("Loaded train data")

    # Read test df
    u_test_data = pd.read_csv(os.path.join(data_path, test_file_name))
    logging.debug("Loaded test data")

    return u_train_data, u_test_data
