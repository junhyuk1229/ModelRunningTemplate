from datetime import datetime
import json
import os
import pandas as pd
import torch


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
    data["raw_test_ratings"] = data["test_ratings"].copy(deep=True)
    data["sample_submission"] = pd.read_csv(
        os.path.join(data_path, "sample_submission.csv")
    )

    return data


class SaveSetting:
    def __init__(self, folder_path, general_settings):
        self.log_folder_path = os.path.join(
            folder_path, general_settings["path"]["log"]
        )
        self.model_folder_path = os.path.join(
            folder_path, general_settings["path"]["model"]
        )
        self.statedict_folder_path = os.path.join(
            folder_path, general_settings["path"]["state_dict"]
        )
        self.submit_folder_path = os.path.join(
            folder_path, general_settings["path"]["submit"]
        )
        self.train_folder_path = os.path.join(
            folder_path, general_settings["path"]["train"]
        )
        self.valid_folder_path = os.path.join(
            folder_path, general_settings["path"]["valid"]
        )
        self.name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = None
        self.create_dir()
        self.start_log(general_settings)

    def create_dir(self) -> None:
        """
        Creates missing directories for save locations
        """
        if not os.path.exists(self.log_folder_path):
            os.mkdir(self.log_folder_path)
        if not os.path.exists(self.model_folder_path):
            os.mkdir(self.model_folder_path)
        if not os.path.exists(self.statedict_folder_path):
            os.mkdir(self.statedict_folder_path)
        if not os.path.exists(self.submit_folder_path):
            os.mkdir(self.submit_folder_path)
        if not os.path.exists(self.train_folder_path):
            os.mkdir(self.train_folder_path)
        if not os.path.exists(self.valid_folder_path):
            os.mkdir(self.valid_folder_path)
        return

    def append_log(self, input_obj) -> None:
        """
        Appends the input string to the log
        """
        if self.log_file is None:
            log_file_path = os.path.join(self.log_folder_path, self.name) + ".txt"
            self.log_file = open(log_file_path, "a")

        self.log_file.writelines(input_obj)

        return

    def start_log(self, general_settings) -> None:
        """
        Starts log by writing initial settings
        """
        with open(os.path.join(self.log_folder_path, self.name) + ".txt", "w") as f:
            f.write("Choosen Columns:\t")
            f.write(", ".join(general_settings["choose_columns"]) + "\n")
            f.write("Indexed Columns:\t")
            f.write(", ".join(general_settings["index_columns"]) + "\n")
            f.write("Model Configuration:\n")
            for index, value in general_settings["run_model"].items():
                f.write(f"\t{index}:\t{value}\n")
            f.write("=" * 30 + "\n\n")
        return

    def close_log(self) -> None:
        """
        Ends log if log is opened
        """
        if self.log_file is not None:
            self.log_file.close()

        return

    def save_model(self, model) -> None:
        """
        Saves model.
        """
        temp_path = os.path.join(self.model_folder_path, self.name)
        temp_path += f"_model"
        torch.save(model, temp_path)

        return

    def save_statedict(self, model, train, valid, settings) -> None:
        """
        Saves model's state dict and extra information
        """
        temp_path = os.path.join(self.statedict_folder_path, self.name)
        temp_path += f"_statedict"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "train": train,
                "valid": valid,
                "settings": settings,
            },
            temp_path,
        )

        return

    def save_submit(self, data, prediction) -> None:
        # Create prediction
        data["raw_test_ratings"]["rating"] = prediction

        # Save prediction
        temp_path = os.path.join(self.submit_folder_path, self.name + ".csv")
        data["raw_test_ratings"].to_csv(temp_path, index=False)

        return

    def save_train_valid(self, train_df, valid_df) -> None:
        # Create path
        train_path = os.path.join(self.train_folder_path, self.name + "_train.csv")
        valid_path = os.path.join(self.valid_folder_path, self.name + "_valid.csv")

        # Save dataframe as csv
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)


def setup() -> tuple[dict, dict, SaveSetting]:
    """
    Returns setting and unprocessed data.

    Returns:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
        save_settings(SaveSetting): Class used to save files(log, model, result).
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
    save_settings = SaveSetting(folder_path, general_settings)

    print("Got Save Settings!")
    print()

    return data, general_settings, save_settings
