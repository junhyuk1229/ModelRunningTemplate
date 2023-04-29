from src.setup import setup
from src.data_process import (
    process_data,
    data_split,
    create_datasets,
    create_dataloader,
)
from src.get_model import create_model
from src.run_model import run_model


def main() -> None:
    # Get settings and raw data from files (getcwd changes to entire folder)
    data, general_settings, save_settings = setup()

    # Process raw data
    process_data(data, general_settings)

    # Split data
    data_split(data, general_settings)

    # Load datasets
    dataset = create_datasets(data, general_settings)

    # Create dataloader
    dataloader = create_dataloader(dataset)

    # Create model
    model = create_model(general_settings)

    # Run model
    predicted_data = run_model(dataloader, general_settings, model, save_settings)

    # Save predicted data as csv
    save_settings.save_submit(data, predicted_data)

    # Close log if opened
    save_settings.close_log()

    return


if __name__ == "__main__":
    main()
