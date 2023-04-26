from .process_age import average_fill_na


def process_mlp(data):
    average_fill_na(data)

    return


def process_data(data, settings):
    print("Merging Data...")

    merged_data = data['train_ratings'].merge(data['user_data'], on='user_id').merge(data['book_data'], on='isbn')

    print("Merged Data!")
    print()

    print("Processing Data...")

    if settings["model"]["name"].lower() == 'mlp':
        process_mlp(merged_data)
    
    merged_data = merged_data[settings['choose_columns']]

    print("Processed Data!")
    print()

    data['processed_data'] = merged_data
