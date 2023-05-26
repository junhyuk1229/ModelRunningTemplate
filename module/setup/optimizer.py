import logging
import torch.optim as optim


def get_optimizer(name: str, params, **kwargs):
    name = name.lower()
    logging.debug(f"Optimizer used: {name}")

    if name == "adam":
        optimizer = optim.Adam(params, kwargs)
    else:
        logging.error(f"No optimizer with name {name}")

    return optimizer
