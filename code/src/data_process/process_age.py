def average_fill_na(data):
    data['age'] = data['age'].fillna(data['age'].mean())

    return