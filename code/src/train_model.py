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


def train_model(dataloader: dict, settings: dict, model) -> None:
    """
    Trains model.

    Parameters:
        dataloader(dict): Dictionary containing the dictionary.
        settings(dict): Dictionary containing the settings.
        model(nn.Module): Model used to train
    """

    # Set loss function
    loss_fn = RMSELoss()
    # Set optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    print("Training Model...")

    # Set epoch for training
    for epoch in range(settings["train"]["epoch"]):
        # Change model state to train
        model.train()

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

        # Print average loss
        average_loss = total_loss / batch_count
        print(f"epoch: {epoch + 1}\tloss: {average_loss.item()}")
    print()

    print("Trained Model!")
    print()

    return
