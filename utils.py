import pickle


# Load Data

def load_data(data_path):

    with open(data_path + 'train_x.p', 'rb') as f:
        train_x = pickle.load(f)
    with open(data_path + 'train_y.p', 'rb') as f:
        train_y = pickle.load(f)
    with open(data_path + 'train_w.p', 'rb') as f:
        train_w = pickle.load(f)

    return train_x, train_y, train_w

