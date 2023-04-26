import json
import os
import pandas as pd


SETTING_FILE = "setting.json"


def get_setting(folder_path):
    with open(os.path.join(folder_path, SETTING_FILE)) as f:
        try:
            settings = json.load(f)
        except:
            print("Running code in wrong folder.")
            print("Run python file in ModelRunningTemplate folder")
            return 1

    return settings


def get_raw_data(data_path):

    data = dict()

    data['book_data'] = pd.read_csv(os.path.join(data_path, 'books.csv'))
    data['user_data'] = pd.read_csv(os.path.join(data_path, 'users.csv'))

    return data


def setup_main():
    os.chdir('..')
    folder_path = os.getcwd()

    print("Getting Settings...")

    settings = get_setting(folder_path)

    print("Loaded Settings!")
    print()
    
    data_path = os.path.join(folder_path, settings["path"]["data"])

    print("Getting Raw Data...")

    data = get_raw_data(data_path)
    
    print("Got Raw Data!")
    print()

    return data, settings
