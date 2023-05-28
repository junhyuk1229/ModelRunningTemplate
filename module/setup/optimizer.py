import logging
import torch
import torch.optim as optim


def get_optimizer_torch(name: str, model, **kwargs) -> torch.optim:
    """
    Gets torch optimizer from name.

    Parameters:
        name(str): The name of the optimizer.
        model(nn.Module): The model using the optimizer.
        **kwargs(dict): The arguments passed to the optimizer.

    Returns:
        optimizer(torch.optim): The optimizer with the models parameters.
    """

    name = name.lower()

    logging.debug(f"Getting optimizer named {name}")

    if name == "adam":
        optimizer = optim.Adam(model.parameters(), **kwargs)
    else:
        logging.error(f"No optimizer with name {name}")

    return optimizer
