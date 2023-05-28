import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


def run_train_torch(
    model: nn.Module,
    optimizer: nn.Module,
    epoch: int,
    loss_fn: nn.Module,
    dataloader: DataLoader,
) -> None:
    """
    Trains model using torch

    Parameters:
        model(nn.Module): Model to be runned.
        optimizer(nn.Module): Optimizer used.
        epoch(int): Amount of epochs ran.
        loss_fn(nn.Module): Loss function used.
        dataloader(DataLoader): Dataloader with data.
    """

    for _ in range(epoch):
        for x, y in dataloader:
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(loss.detach())

    return


def run_train_sklearn(model, x, y) -> float:
    """
    Trains model using sklearn.

    Parameters:
        model: Model used to train.
        x: Train input values.
        y: Train output values.

    Returns:
        acc(float): Accuracy of model.
    """

    model.fit(x, y)

    acc = model.score(x, y)

    return acc


def run_test_sklearn(model, x) -> np.ndarray:
    """
    Trains model using sklearn.

    Parameters:
        model: Model used to train.
        x: Test input values.

    Returns:
        y_hat(np.ndarray): Test output values.
    """

    y_hat = model.predict(x)

    return y_hat
