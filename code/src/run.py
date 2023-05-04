import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.functional import sigmoid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup


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
    elif settings["run_model"]["loss_fn"].lower() == "bcewll":
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Set optimizer
    if settings["run_model"]["optimizer"].lower() == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=settings["run_model"]["learn_rate"],
            weight_decay=0.01,
        )

    scheduler = ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, mode="max", verbose=True
    )

    print("Training Model...")
    print()

    best_auc = -1

    # Set epoch for training
    for epoch in range(settings["run_model"]["epoch"]):
        # Change model state to train
        model.train()

        # Get average loss while training
        train_auc, train_acc = train_model(
            dataloader, model, loss_fn, optimizer, scheduler
        )

        # Change model state to evaluation
        model.eval()

        # Get average loss using validation set
        valid_acc, valid_auc = validate_model(dataloader, model, loss_fn)

        if valid_auc > best_auc:
            best_auc = valid_auc

        scheduler.step(best_auc)

        # Print average loss of train/valid set
        print(
            f"Epoch: {epoch + 1}\nTrain acc: {train_acc}\tTrain auc: {train_auc}\nValid acc: {valid_acc}\t Valid auc: {valid_auc}\n"
        )

        save_settings.append_log(
            f"Epoch: {epoch + 1}\nTrain acc: {train_acc}\tTrain auc: {train_auc}\nValid acc: {valid_acc}\t Valid auc: {valid_auc}"
        )

    print()

    print("Trained Model!")
    print()

    return

    print("Getting Final Results...")

    # Get final results
    train_df, train_final_loss = get_df_result(
        dataloader["train_dataloader"], model, loss_fn
    )
    valid_df, valid_final_loss = get_df_result(
        dataloader["valid_dataloader"], model, loss_fn
    )

    save_settings.save_train_valid(train_df, valid_df)

    print(
        f"Final results:\tTrain loss: {train_final_loss}\tValid loss: {valid_final_loss}"
    )

    print("Got Final Results!")
    print()

    print("Saving Model/State Dict...")

    # Save model and state_dict, loss, settings
    save_settings.save_model(model)
    save_settings.save_statedict(model, train_final_loss, valid_final_loss, settings)

    print("Saved Model/State Dict!")
    print()

    print("Predicting Results...")

    # Get predicted data for submission
    predict_data = test_model(dataloader, model)

    print("Predicted Results!")
    print()

    return predict_data


def train_model(dataloader: dict, model, loss_fn, optimizer, scheduler) -> float:
    """
    Trains model.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train
        loss_fn: Used to find the loss between two tensors
        optimizer: Used to optimize parameters
    """

    total_preds = []
    total_targets = []
    losses = []

    for data in dataloader["train"]:
        # Split data to input and output
        x = data
        y = data["answerCode"]

        # Get predicted output with input
        y_hat = model(x)

        # Get loss using predicted output
        loss = loss_fn(y_hat, y.float())

        loss = loss[:, -1]
        loss = torch.mean(loss)

        # Computes the gradient of current parameters
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 10)

        # Optimize parameters
        optimizer.step()

        # Set the gradients of all optimized parameters to zero
        optimizer.zero_grad()

        y_hat = sigmoid(y_hat[:, -1])
        y = y[:, -1]

        total_preds.append(y_hat.detach())
        total_targets.append(y.detach())
        losses.append(loss.detach())

    total_targets = torch.concat(total_targets).numpy()
    total_preds = torch.concat(total_preds).numpy()

    auc = roc_auc_score(y_true=total_targets, y_score=total_preds)
    acc = accuracy_score(
        y_true=total_targets, y_pred=np.where(total_preds >= 0.5, 1, 0)
    )

    return auc, acc


def validate_model(dataloader: dict, model, loss_fn) -> float:
    """
    Uses valid dataloader to get loss of model.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        model(nn.Module): Model used to train
        loss_fn: Used to find the loss between two tensors
    """

    total_preds = []
    total_targets = []
    losses = []

    # No learning from validation data
    with torch.no_grad():
        for data in dataloader["valid"]:
            # Split data to input and output
            x = data
            y = data["answerCode"]

            # Get predicted output with input
            y_hat = model(x)

            # Get loss using predicted output
            loss = loss_fn(y_hat, y.float())

            loss = loss[:, -1]
            loss = torch.mean(loss)

            y_hat = sigmoid(y_hat[:, -1])
            y = y[:, -1]

            total_preds.append(y_hat.detach())
            total_targets.append(y.detach())
            losses.append(loss.detach())

    total_targets = torch.concat(total_targets).numpy()
    total_preds = torch.concat(total_preds).numpy()

    auc = roc_auc_score(y_true=total_targets, y_score=total_preds)
    acc = accuracy_score(
        y_true=total_targets, y_pred=np.where(total_preds >= 0.5, 1, 0)
    )

    return auc, acc


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


def get_df_result(dataloader, model, loss_fn):
    """


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

    # Create dataframe to save
    column_list = ["user_id", "isbn", "rating", "p_rating"]
    save_df = pd.DataFrame({c: [] for c in column_list})

    for data in dataloader:
        # Split data to input and output
        x, y = data

        # Get predicted output with input
        y_hat = model(x)

        # Update dataframe
        x = pd.DataFrame(x, columns=["user_id", "isbn", "age", "year"])
        x = x[["user_id", "isbn"]]
        x["rating"] = y
        x["p_rating"] = y_hat.detach().numpy()
        save_df = pd.concat([save_df, x])

        # Get loss using predicted output
        loss = loss_fn(y, torch.squeeze(y_hat))

        # Get cumulative loss and count
        total_loss += loss.clone().detach()
        batch_count += 1

    # Get average loss
    average_loss = total_loss / batch_count

    return save_df, average_loss.item()
