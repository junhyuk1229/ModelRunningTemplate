import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset


def data_split(data: dict, settings: dict) -> None:
    """
    Splits train data to train data and validation data.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """
    print("Splitting dataset...")

    # Split data to valid and test
    X_train, X_valid, y_train, y_valid = train_test_split(
        data["train_ratings"].drop([settings["predict_column"]], axis=1),
        data["train_ratings"][settings["predict_column"]],
        test_size=0.9,
        random_state=42,
        shuffle=True,
    )

    # Save data to dict
    data["X_train"] = X_train
    data["X_valid"] = X_valid
    data["y_train"] = y_train
    data["y_valid"] = y_valid

    print("Splitted Data!")
    print()

    return


def load_datasets(data: dict, settings: dict) -> dict:
    """
    Splits train data to train data and validation data.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.

    Returns:
        dataset_dict(dict): Dictionary loaded datasets.
    """

    print("Loading Datasets!")

    dataset = dict()

    train_dataset = TensorDataset(
        torch.LongTensor(data["X_train"].values),
        torch.LongTensor(data["y_train"].values),
    )

    valid_dataset = TensorDataset(
        torch.LongTensor(data["X_valid"].values),
        torch.LongTensor(data["y_valid"].values),
    )

    test_dataset = TensorDataset(torch.LongTensor(data["test_ratings"].values))

    dataset["train_dataset"] = train_dataset
    dataset["valid_dataset"] = valid_dataset
    dataset["test_dataset"] = test_dataset

    print("Loaded Datasets!")
    print()

    return dataset
