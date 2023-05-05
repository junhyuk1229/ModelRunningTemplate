from src.setup import setup
from src.data import (
    process_data,
    data_split,
    create_datasets,
    create_dataloader,
)
from src.model import create_model
from src.run import run_model


def main() -> None:
    # Get settings and raw data from files (os.getcwd changes to entire folder)
    data, settings, save_settings = setup()

    # Process raw data
    process_data(data, settings)

    # Split data
    data_split(data, settings)

    # Load datasets
    dataset = create_datasets(data, settings)

    # Create dataloader
    dataloader = create_dataloader(dataset, settings)

    # Create model
    model = create_model(data, settings)

    # Run model
    predicted_data = run_model(dataloader, settings, model, save_settings)

    # Save predicted data as csv
    save_settings.save_submit(predicted_data)

    # Close log if opened
    save_settings.close_log()

    return


if __name__ == "__main__":
    main()
