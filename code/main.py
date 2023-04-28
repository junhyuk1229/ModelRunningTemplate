from src.setup import setup
from src.data_process import (
    process_data,
    data_split,
    create_datasets,
    create_dataloader,
)
from src.get_model import create_model
from src.train_model import run_model


def main() -> None:
    # Get settings and raw data from files (getcwd changes to entire folder)
    data, settings = setup()

    # Process raw data
    process_data(data, settings)

    # Split data
    data_split(data, settings)

    # Load datasets
    dataset = create_datasets(data, settings)

    # Create dataloader
    dataloader = create_dataloader(dataset)

    # Create model
    model = create_model(settings)

    # Train model
    run_model(dataloader, settings, model)

    return


if __name__ == "__main__":
    main()
