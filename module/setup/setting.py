import json
import logging
import numpy as np
import os
import pandas as pd
import random
import torch


def get_setting(setting_file_path: str) -> dict:
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

    logging.debug("Loaded All Settings")

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
    logging.debug("Seeded all data")


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


def get_unprocessed_data(
    data_path: str, train_file_name: str, test_file_name: str
) -> dict:
    """
    Gets unprocessed data as dataframe and returns it in a dictionary.

    Parameters:
        data_path(str): The path to the data folder.
        train_file_name(str): The file name of the train csv file
        test_file_name(str): The file name of the test csv file

    Returns:
        u_train_data(pd.DataFrame): Dictionary containing the unprocessed train dataframes.
        u_test_data(pd.DataFrame): Dictionary containing the unprocessed test dataframes.
    """

    # Read train df
    u_train_data = pd.read_csv(os.path.join(data_path, train_file_name))
    logging.debug("Loaded train data")

    # Read test df
    u_test_data = pd.read_csv(os.path.join(data_path, test_file_name))
    logging.debug("Loaded test data")

    return u_train_data, u_test_data
