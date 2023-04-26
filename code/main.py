from src.setup import setup
from src.data_process import process_data


def main():
    data, settings = setup()

    process_data(data, settings)

    return 0


if __name__ == '__main__':
    main()
