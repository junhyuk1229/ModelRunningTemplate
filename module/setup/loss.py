import logging
import torch.nn as nn


def get_loss_function_torch(name: str) -> nn.Module:
    """
    Gets torch loss function from name.

    Parameters:
        name(str): Name of the loss function.

    Returns:
        loss_fn(nn.Module): Loss function.
    """

    name = name.lower()

    logging.debug(f"Getting loss function named {name}")

    # Get loss function from name
    if name == "mse":
        loss_fn = nn.MSELoss()
    else:
        logging.error(f"No loss function with name {name}")

    return loss_fn
