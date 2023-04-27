from src.setup import setup
from src.data_process import process_data, data_split, load_datasets


def main() -> None:
    # Get settings and raw data from files (getcwd changes to entire folder)
    data, settings = setup()

    # Process raw data
    process_data(data, settings)

    # Split data
    data_split(data, settings)

    # Load datasets
    dataset = load_datasets(data, settings)

    print(dataset)

    return


if __name__ == "__main__":
    main()
