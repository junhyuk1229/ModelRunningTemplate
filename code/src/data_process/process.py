def process_mlp(data, settings):
    return data


def process_data(data, settings):
    print("Processing Data...")

    if settings["model"]["name"].lower() == 'mlp':
        process_mlp(data, settings)

    print("Processed Data!")
    print()

    print("Merging Data...")

    merged_data = data['train_ratings'].merge(data['user_data'], on='user_id').merge(data['book_data'], on='isbn')

    print("Merged Data!")

    data['merged_data'] = merged_data
