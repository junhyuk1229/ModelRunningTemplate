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

    return dataloader
