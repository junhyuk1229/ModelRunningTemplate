import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


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


def create_datasets(data: dict, settings: dict) -> dict:
    """
    Creates datasets using the merged train, valid, test data.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.

    Returns:
        dataset(dict): Dictionary loaded datasets.
    """

    print("Creating Datasets!")

    dataset = dict()

    train_dataset = TensorDataset(
        torch.FloatTensor(data["X_train"].values),
        torch.FloatTensor(data["y_train"].values),
    )

    valid_dataset = TensorDataset(
        torch.FloatTensor(data["X_valid"].values),
        torch.FloatTensor(data["y_valid"].values),
    )

    test_dataset = TensorDataset(torch.FloatTensor(data["test_ratings"].values))

    dataset["train_dataset"] = train_dataset
    dataset["valid_dataset"] = valid_dataset
    dataset["test_dataset"] = test_dataset

    print("Created Datasets!")
    print()

    return dataset


def create_dataloader(dataset: dict):
    """
    Creates dataloader from datasets.

    Parameters:
        dataset(dict): Dictionary containing datasets.

    Returns:
        dataloader(dict): Dictionary loaded datasets.
    """

    print("Creating Dataloader...")

    dataloader = dict()

    train_dataloader = DataLoader(dataset["train_dataset"], batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(dataset["valid_dataset"], batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset["test_dataset"], batch_size=32, shuffle=False)

    dataloader["train_dataloader"] = train_dataloader
    dataloader["valid_dataloader"] = valid_dataloader
    dataloader["test_dataloader"] = test_dataloader

    print("Created Dataloader!")
    print()

    return dataloader
