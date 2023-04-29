import numpy as np
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


def run_model(dataloader: dict, settings: dict, model, save_settings):
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

    if settings["run_model"]["name"].lower() == "mlp":
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

            # Change model state to evaluation
            model.eval()

            # Get average loss using validation set
            valid_average_loss = validate_model(dataloader, model, loss_fn)

            # Print average loss of train/valid set
            print(
                f"Epoch: {epoch + 1}\tTrain loss: {train_average_loss}\tValid loss: {valid_average_loss}"
            )

            save_settings.append_log(
                f"Epoch: {epoch + 1}\tTrain loss: {train_average_loss}\tValid loss: {valid_average_loss}\n"
            )

        print()

    elif settings["run_model"]["name"].lower() == "mf":
        # Create R matrix
        R = np.zeros((settings["user_num"], settings["book_num"]), )
        
        for data in dataloader["train_dataloader"]:
            x, y = data
            user, book = x
            R[user, book] = y
        
        print(R)
        # Find error
        # Update parameters
        # Check valid error
        pass

    print("Trained Model!")
    print()

    print("Saving Model/State Dict...")

    # Save model and state_dict, loss, settings
    save_settings.save_model(model)
    save_settings.save_statedict(
        model, valid_average_loss, train_average_loss, settings
    )

    print("Saved Model/State Dict!")
    print()

    print("Predicting Results...")

    # Get predicted data for submission
    predict_data = test_model(dataloader, model)

    print("Predicted Results!")
    print()

    return predict_data


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

    # No learning from validation data
    with torch.no_grad():
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


def test_model(dataloader: dict, model) -> list:
    """
    Use test data to get prediction for submission.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train

    Returns:
        predicted_list(list): Predicted results from test dataset.
    """
    # Predicted values in order
    predicted_list = list()

    with torch.no_grad():
        for data in dataloader["test_dataloader"]:
            # Get input data
            x = data[0]

            # Get predicted output with input
            y_hat = model(x)

            # Add predicted output to list
            predicted_list.extend(y_hat.squeeze().tolist())

    return predicted_list
