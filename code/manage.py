import os
import pandas as pd
import shutil
from src.setup import get_general_setting
import torch
import torch.nn as nn
from torch.nn import MSELoss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss


def get_df(statedict_path):
    # Get data from statedict
    if os.path.exists(statedict_path):
        statedict_path_list = os.listdir(statedict_path)
    else:
        print("No Results Detected...")
        print("Ending Program")

        return None

    train_acc_list = []
    train_auc_list = []
    valid_acc_list = []
    valid_auc_list = []
    model_name = []
    settings = []
    for p in statedict_path_list:
        temp_dict = torch.load(os.path.join(statedict_path, p))
        train_acc_list.append(temp_dict["train_acc"])
        train_auc_list.append(temp_dict["train_auc"])
        valid_acc_list.append(temp_dict["valid_acc"])
        valid_auc_list.append(temp_dict["valid_auc"])
        model_name.append(temp_dict["settings"]["model_name"].lower())
        settings.append(temp_dict["settings"])

    statedict_path_list = [p[:15] for p in statedict_path_list]

    result_df = pd.DataFrame(statedict_path_list, columns=["file_name"])

    result_df["model_name"] = model_name
    result_df["train_acc_list"] = train_acc_list
    result_df["train_auc_list"] = train_auc_list
    result_df["valid_acc_list"] = valid_acc_list
    result_df["valid_auc_list"] = valid_auc_list

    return result_df, settings


def clear_data(path_list):
    print("Are you sure?(type 'confirm'): ", end="")

    input_str = input()

    if input_str == "confirm":
        for p in path_list:
            if os.path.exists(p):
                shutil.rmtree(p)
        print("Cleared all results!")
        return True
    else:
        return False


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

    ending_str = [".txt", "_model", "_statedict", ".csv", "_train.csv", "_valid.csv"]

    for p, e_s in zip(path_list, ending_str):
        os.remove(os.path.join(p, file_name + e_s))

    result_df.drop(index=int(input_str[1]), inplace=True)

    return


def ensemble(result_df, input_str, path_list, ensemble_path):
    select_rows = list(map(int, input_str[1].split(sep=",")))
    ensemble_name = result_df["file_name"][select_rows].T.values
    train_acc_list = result_df["train_acc_list"][select_rows].T.values
    train_auc_list = result_df["train_auc_list"][select_rows].T.values
    valid_acc_list = result_df["valid_acc_list"][select_rows].T.values
    valid_auc_list = result_df["valid_auc_list"][select_rows].T.values

    train_df = []
    valid_df = []
    for file_name in ensemble_name:
        train_path = os.path.join(path_list[-2], file_name) + "_train.csv"
        valid_path = os.path.join(path_list[-1], file_name) + "_valid.csv"
        train_df.append(pd.read_csv(train_path))
        valid_df.append(pd.read_csv(valid_path))

    print(train_df)

    train_df[0][0]

    return

    if len(input_str) < 3:
        weight_list = [1 / len(select_rows)] * len(select_rows)
    else:
        weight_list = list(map(float, input_str[2].split(sep=",")))
        total_sum = sum(weight_list)
        weight_list = [f / total_sum for f in weight_list]

    for i, w in enumerate(weight_list):
        temp_train = train_df[i].apply(lambda x: x * w)
        temp_valid = valid_df[i].apply(lambda x: x * w)

        if i == 0:
            train_total = temp_train
            valid_total = temp_valid
        else:
            train_total = train_total.add(temp_train)
            valid_total = valid_total.add(temp_valid)

    loss = RMSELoss()
    train_loss = loss(
        torch.tensor(train_total.values),
        torch.tensor(pd.read_csv(train_path)["rating"].values),
    )
    valid_loss = loss(
        torch.tensor(valid_total.values),
        torch.tensor(pd.read_csv(valid_path)["rating"].values),
    )

    os.system("cls" if os.name == "nt" else "clear")
    print(f"File train loss: {train_list}\tEnsemble train loss: {train_loss}")
    print(f"File valid loss: {valid_list}\tEnsemble valid loss: {valid_loss}")
    print("Do you want to make the file?(y): ")

    response = input()
    if response == "y":
        if not os.path.exists(ensemble_path):
            os.mkdir(ensemble_path)

        test_df = []
        for file_name in ensemble_name:
            test_path = os.path.join(path_list[-3], file_name) + ".csv"
            test_df.append(pd.read_csv(test_path)["rating"])

        output_df = pd.read_csv(test_path).drop(["rating"], axis=1)

        for i, w in enumerate(weight_list):
            temp_test = test_df[i].apply(lambda x: x * w)

            if i == 0:
                test_total = temp_test
            else:
                test_total = test_total.add(temp_test)

        output_df["rating"] = test_total.values

        output_df.to_csv(
            os.path.join(
                ensemble_path,
                f"{train_loss:.3f}_{valid_loss:.3f}_{'_'.join(ensemble_name)}",
            )
        )

    return


def main() -> None:
    os.chdir("..")
    folder_path = os.getcwd()

    settings = get_general_setting(folder_path)

    log_path = os.path.join(folder_path, settings["path"]["log"])
    model_path = os.path.join(folder_path, settings["path"]["model"])
    statedict_path = os.path.join(folder_path, settings["path"]["state_dict"])
    submit_path = os.path.join(folder_path, settings["path"]["submit"])
    train_path = os.path.join(folder_path, settings["path"]["train"])
    valid_path = os.path.join(folder_path, settings["path"]["valid"])
    ensemble_path = os.path.join(folder_path, settings["path"]["ensemble"])

    path_list = [
        log_path,
        model_path,
        statedict_path,
        submit_path,
        train_path,
        valid_path,
    ]

    result_df, settings = get_df(statedict_path)

    if result_df is None:
        return

    origin_df = result_df.copy(deep=True)

    head_num = 5

    while True:
        os.system("cls" if os.name == "nt" else "clear")

        if len(result_df) == 0:
            print("Detected empty dataframe...")
            print("Restored original dataframe")
            result_df = origin_df.copy(deep=True)

        print(f"Printing First {head_num} data...")
        print(result_df[:head_num])
        print()
        print("Commands: clear, sort, model, reload, delete, head, exit")
        print("Enter your command: ", end="")

        input_str = input().split(sep=" ")

        if input_str[0] == "clear":
            if clear_data(path_list):
                break
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
        elif input_str[0] == "head":
            head_num = int(input_str[1])
        elif input_str[0] == "ensemble":
            ensemble(result_df, input_str, path_list, ensemble_path)
        elif input_str[0] == "settings":
            print(settings[int(input_str[1])])
            input()
        else:
            print("Unrecognised command...\nExiting...\n")
            break
    return


if __name__ == "__main__":
    main()
