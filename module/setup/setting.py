import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import torch


def get_json_setting(setting_file_path: str) -> dict:
    """
    Returns the setting.json file as a dictionary.

    Parameters:
        setting_file_path(str): The path to the json file.

    Returns:
        settings(dict): Dictionary containing the settings.
    """

    # Setting file name
    SETTING_FILE = "setting.json"

    with open(os.path.join(setting_file_path, SETTING_FILE)) as f:
        settings = json.load(f)

    logging.debug("Loaded All Settings from JSON File")

    return settings


def check_cuda(used_device: str) -> None:
    """
    Checks if cuda is available and changes device accordingly

    Parameters:
        used_device(str): The device used for training.

    Returns:
        used_device(str): The device used for training.
    """

    # Check if GPU is available
    if not torch.cuda.is_available() and used_device == "cuda":
        logging.warning("No cuda device detected, changing device to cpu")
        # If not change device to cpu
        used_device = "cpu"

    return used_device


def seed_program(seed: int) -> None:
    """
    Sets seeds to remove random chance

    Parameters:
        seed(int): The integer used to set seed
    """

    # Apply settings to all randomization
    # All results will be fixed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.debug("Set all seeds")


def create_logger(
    log_path: str,
    logging_level: int,
    do_log_file: bool = True,
    do_log_print: bool = True,
) -> None:
    """
    Sets logger using path and logging level.

    Parameters:
        log_path(str): Path to logging file
        logging_level(int): Level of logging needed
    """

    # Used to set the logging level
    LOGGING_LEVEL_LIST = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]

    # Get handlers
    log_handler = []

    if do_log_file:
        # Handles file logging
        log_handler.append(logging.FileHandler(filename=log_path, mode="a"))
    if do_log_print:
        # Handles print logging
        log_handler.append(logging.StreamHandler())

    # Set logging configuration
    logging.basicConfig(
        level=LOGGING_LEVEL_LIST[logging_level],
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=log_handler,
    )

    logging.debug("Created Logger")

    return


def get_argparse_settings() -> argparse.Namespace:
    """
    Get settings from argparse

    Returns:
        args(argparse.Namespace): A namespace with all the settings in args.
    """

    parser = argparse.ArgumentParser(description="Describe the program")

    parser.add_argument(
        "-lr",
        "--learn_rate",
        help="learn rate of the optimizer",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-ll",
        "--log_level",
        help="Sets the level of logging",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=3,
    )

    args = parser.parse_args()

    logging.debug(f"Set learn_rate to {args.learn_rate}")
    logging.debug(f"Set logging to {args.log_level}")

    logging.debug(f"Loaded All Settings from Parseargs")

    return args
