import os
import pandas as pd
import shutil
from src.setup import get_general_setting
import sys
import torch


def get_df(statedict_path):
    # Get data from statedict
    if os.path.exists(statedict_path):
        statedict_path_list = os.listdir(statedict_path)
    else:
        print("No Results Detected...")
        print("Ending Program")

        return

    train_list = []
    valid_list = []
    model_name = []
    for p in statedict_path_list:
        temp_dict = torch.load(os.path.join(statedict_path, p))
        train_list.append(temp_dict["train"])
        valid_list.append(temp_dict["valid"])
        model_name.append(temp_dict["settings"]["run_model"]["name"].lower())

    statedict_path_list = [p[:15] for p in statedict_path_list]

    result_df = pd.DataFrame(statedict_path_list, columns=["file_name"])

    result_df["train_loss"] = train_list
    result_df["valid_loss"] = valid_list
    result_df["model_name"] = model_name

    return result_df


def clear_data(path_list):
    print("Are you sure?(type 'confirm'): ", end="")

    input_str = input()

    if input_str == "confirm":
        for p in path_list:
            if os.path.exists(p):
                shutil.rmtree(p)
        return
    else:
        return


def sort_df(result_df, input_str):
    if len(input_str) < 2:
        return

    if len(input_str) < 3:
        result_df.sort_values(by=input_str[1], inplace=True)
    else:
        ascending = True
        if input_str[2].lower() == "false":
            ascending = False
        result_df.sort_values(by=input_str[1], ascending=ascending, inplace=True)

    return


def choose_model(result_df, input_str):
    if len(input_str) < 2:
        return result_df

    result_df = result_df[result_df["model_name"] == input_str[1].lower()]
    return result_df


def delete_model(result_df, input_str, path_list):
    if len(input_str) < 2:
        return

    file_name = result_df["file_name"][int(input_str[1])]

    print(file_name)

    ending_str = [".txt", "_model", "_statedict", ".csv"]

    for p, e_s in zip(path_list, ending_str):
        os.remove(os.path.join(p, file_name + e_s))

    result_df.drop(index=int(input_str[1]), inplace=True)

    return


def main() -> None:
    os.chdir("..")
    folder_path = os.getcwd()

    settings = get_general_setting(folder_path)

    log_path = os.path.join(folder_path, settings["path"]["log"])
    model_path = os.path.join(folder_path, settings["path"]["model"])
    statedict_path = os.path.join(folder_path, settings["path"]["state_dict"])
    submit_path = os.path.join(folder_path, settings["path"]["submit"])

    path_list = [log_path, model_path, statedict_path, submit_path]

    result_df = get_df(statedict_path)
    origin_df = result_df.copy(deep=True)

    while True:
        os.system("cls" if os.name == "nt" else "clear")

        if len(result_df) == 0:
            print("Detected empty dataframe...")
            print("Restored original dataframe")
            result_df = origin_df.copy(deep=True)

        print("Printing First 5 data...")
        print(result_df[:5])
        print()
        print("Commands: clear, sort, model, reload, exit")
        print("Enter your command: ", end="")

        input_str = input().split(sep=" ")

        if input_str[0] == "clear":
            clear_data(path_list)
        elif input_str[0] == "sort":
            sort_df(result_df, input_str)
        elif input_str[0] == "model":
            result_df = choose_model(result_df, input_str)
        elif input_str[0] == "reload":
            result_df = origin_df.copy(deep=True)
        elif input_str[0] == "delete":
            delete_model(result_df, input_str, path_list)
            origin_df = result_df.copy(deep=True)
        elif input_str[0] == "exit":
            print("Exiting...\n")
            break
        else:
            print("Unrecognised command...\nExiting...\n")
            break

    return


if __name__ == "__main__":
    main()
