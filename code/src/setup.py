import json
import os
import pandas as pd


# Setting file name
SETTING_FILE = "setting.json"


def get_general_setting(folder_path: str) -> dict:
    """
    Returns the setting.json file as a dictionary.

    Parameters:
        folder_path(str): The path to the json file.

    Returns:
        settings(dict): Dictionary containing the settings.
    """

    with open(os.path.join(folder_path, SETTING_FILE)) as f:
        try:
            settings = json.load(f)
        except:
            print("Running code in wrong folder.")
            print("Run python file in ModelRunningTemplate folder")
            return 1

    return settings


def get_unprocessed_data(data_path: str) -> dict:
    """
    Gets unprocessed data as dataframe and returns it in a dictionary.

    Parameters:
        data_path(str): The path to the data folder.

    Returns:
        data(dict): Dictionary containing the unprocessed dataframes.
    """

    data = dict()

    # Read csv files
    data["book_data"] = pd.read_csv(os.path.join(data_path, "books.csv"))
    data["user_data"] = pd.read_csv(os.path.join(data_path, "users.csv"))
    data["train_ratings"] = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    data["test_ratings"] = pd.read_csv(os.path.join(data_path, "test_ratings.csv"))
    data["sample_submission"] = pd.read_csv(
        os.path.join(data_path, "sample_submission.csv")
    )

    return data


class SaveSetting():
    def __init__(self, folder_path, general_settings):
        self.log_folder_path = os.path.join(folder_path, general_settings['path']['log'])
        self.create_dir()

    def create_dir(self):
        '''
        Creates missing directories for save locations
        '''
        if not os.path.exists(self.log_folder_path):
            os.mkdir(self.log_folder_path)

        return


def setup() -> tuple[dict, dict, SaveSetting]:
    """
    Returns setting and unprocessed data.

    Returns:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """

    # Changes directory to parent directory
    os.chdir("..")
    folder_path = os.getcwd()

    print("Getting General Settings...")

    # Import settings
    general_settings = get_general_setting(folder_path)

    print("Loaded General Settings!")
    print()

    # Data path to data file
    data_path = os.path.join(folder_path, general_settings["path"]["data"])

    print("Getting Unprocessed Data...")

    # Import data
    data = get_unprocessed_data(data_path)

    print("Got Unprocessed Data!")
    print()

    print("Getting Save Settings...")

    # Get save data
    save_class = SaveSetting(folder_path, general_settings)

    print("Got Save Settings!")
    print()

    return data, general_settings, save_class
