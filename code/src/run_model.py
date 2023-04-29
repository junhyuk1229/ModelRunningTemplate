import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss


def run_model(dataloader: dict, settings: dict, model) -> None:
    """
    Runs model through train, valid, and submit.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        settings(dict): Dictionary containing the settings.
        model(nn.Module): Model used to train
    """

    # Set loss function
    if settings["run_model"]["loss_fn"].lower() == "rmse":
        loss_fn = RMSELoss()
    elif settings["run_model"]["loss_fn"].lower() == "mse":
        loss_fn = MSELoss()

    # Set optimizer
    if settings["run_model"]["optimizer"].lower() == "adam":
        optimizer = Adam(model.parameters(), lr=settings["run_model"]["learn_rate"])

    print("Training Model...")
    print()

    # Set epoch for training
    for epoch in range(settings["run_model"]["epoch"]):
        # Change model state to train
        model.train()

        # Get average loss while training
        train_average_loss = train_model(dataloader, model, loss_fn, optimizer)

        model.eval()

        valid_average_loss = validate_model(dataloader, model, loss_fn)

        # Print average loss
        print(
            f"Epoch: {epoch + 1}\tTrain loss: {train_average_loss}\tValid loss: {valid_average_loss}"
        )
    print()

    print("Trained Model!")
    print()

    return


def train_model(dataloader: dict, model, loss_fn, optimizer) -> float:
    """
    Trains model.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train
        loss_fn: Used to find the loss between two tensors
        optimizer: Used to optimize parameters
    """

    # Total sum of loss
    total_loss = 0

    # Number of batches trained
    batch_count = 0

    for data in dataloader["train_dataloader"]:
        # Split data to input and output
        x, y = data

        # Get predicted output with input
        y_hat = model(x)

        # Get loss using predicted output
        loss = loss_fn(y, torch.squeeze(y_hat))

        # Set the gradients of all optimized parameters to zero
        optimizer.zero_grad()

        # Computes the gradient of current parameters
        loss.backward()

        # Optimize parameters
        optimizer.step()

        # Get cumulative loss and count
        total_loss += loss.clone().detach()
        batch_count += 1

    average_loss = total_loss / batch_count

    return average_loss.item()


def validate_model(dataloader: dict, model, loss_fn) -> float:
    """
    Uses valid dataloader to get loss of model.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train
        loss_fn: Used to find the loss between two tensors
    """

    # Total sum of loss
    total_loss = 0

    # Number of batches trained
    batch_count = 0

    for data in dataloader["valid_dataloader"]:
        # Split data to input and output
        x, y = data

        # Get predicted output with input
        y_hat = model(x)

        # Get loss using predicted output
        loss = loss_fn(y, torch.squeeze(y_hat))

        # Get cumulative loss and count
        total_loss += loss
        batch_count += 1

    average_loss = total_loss / batch_count

    return average_loss.item()
