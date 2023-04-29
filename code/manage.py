import os
import shutil
from src.setup import get_general_setting
import sys


def main() -> None:
    os.chdir("..")
    folder_path = os.getcwd()

    settings = get_general_setting(folder_path)

    log_path = os.path.join(folder_path, settings["path"]["log"])
    model_path = os.path.join(folder_path, settings["path"]["model"])
    statedict_path = os.path.join(folder_path, settings["path"]["state_dict"])
    submit_path = os.path.join(folder_path, settings["path"]["submit"])

    while True:
        print("Enter your command: ", end="")

        input_str = sys.stdin.readline().rstrip()

        if input_str == "clear":
            print("Are you sure?(y/n): ")

            input_str = sys.stdin.readline().rstrip()

            if input_str.lower() == "y":
                shutil.rmtree(log_path)
                shutil.rmtree(model_path)
                shutil.rmtree(statedict_path)
                shutil.rmtree(submit_path)
            else:
                continue
        elif input_str == "exit":
            print("Exiting...\n")
            break
        else:
            print("Unrecognised command...\nExiting...\n")
            break

    return


if __name__ == "__main__":
    main()
