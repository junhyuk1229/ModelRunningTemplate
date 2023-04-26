from src.setup import setup
from src.data_process import process_data


def main() -> None:
    # Get settings and raw data from files (getcwd changes to entire folder)
    data, settings = setup()

    # Process raw data
    process_data(data, settings)

    return


if __name__ == "__main__":
    main()
