import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, settings: dict):
        self.data = data
        self.max_seq_len = settings["max_seq_len"]
        self.predict_column = settings["predict_column"]

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        # Load from data
        data = {
            i: torch.tensor(v + 1, dtype=torch.int)
            for i, v in row.items()
            if i != "answerCode"
        }
        data["answerCode"] = torch.tensor(row["answerCode"], dtype=torch.int)

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(list(row.values())[0])

        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len :]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len - seq_len :] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1

        data["mask"] = mask

        # Generate interaction
        interaction = data[self.predict_column] + 1
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}

        return data

    def __len__(self) -> int:
        return len(self.data)


def data_split(data: dict, settings: dict) -> None:
    """
    Splits train data to train data and validation data.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """
    print("Splitting dataset...")

    # Group by user
    column_list = data["train"].columns
    data["train"] = (
        data["train"]
        .groupby("userID")
        .apply(lambda x: {c: x[c].values for c in column_list if c != "userID"})
    )
    data["test"] = (
        data["test"]
        .groupby("userID")
        .apply(lambda x: {c: x[c].values for c in column_list if c != "userID"})
    )

    data["train"] = data["train"].to_numpy()
    data["test"] = data["test"].to_numpy()

    # Split data to valid and test
    random.seed(0)  # fix to default seed 0
    random.shuffle(data["train"])

    size = int(len(data["train"]) * 0.7)
    data["valid"] = data["train"][size:]
    data["train"] = data["train"][:size]

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

    dataset["train"] = DKTDataset(data["train"], settings)
    dataset["valid"] = DKTDataset(data["valid"], settings)
    dataset["test"] = DKTDataset(data["test"], settings)

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

    dataloader["train"] = DataLoader(
        dataset["train"], batch_size=64, shuffle=True, num_workers=1, pin_memory=False
    )
    dataloader["valid"] = DataLoader(
        dataset["valid"], batch_size=64, shuffle=False, num_workers=1, pin_memory=False
    )
    dataloader["test"] = DataLoader(
        dataset["test"], batch_size=64, shuffle=False, num_workers=1, pin_memory=False
    )

    print("Created Dataloader!")
    print()

    return dataloader
