import pickle


# Save Data

def save_data(data, data_path):

    print('Saving ' + data_path + '...')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


# Load Data

def load_data(data_path):

    print('Loading ' + data_path + '...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


# Load Preprocessed Data

def load_preprocessed_data(data_file_path):

    print('Loading preprocessed data...')

    train_x = load_data(data_file_path + 'train_x.p')
    train_y = load_data(data_file_path + 'train_y.p')
    train_w = load_data(data_file_path + 'train_w.p')

    return train_x, train_y, train_w

