import pickle
import pandas as pd
import numpy as np


# Save Data

def save_np_to_pkl(data, data_path):

    print('Saving ' + data_path + '...')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


# Load Data

def load_pkl_to_np(data_path):

    print('Loading ' + data_path + '...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


# Load Preprocessed Data

def load_preprocessed_np_data(data_file_path):

    print('Loading preprocessed data...')

    train_x = load_pkl_to_np(data_file_path + 'train_x.p')
    train_y = load_pkl_to_np(data_file_path + 'train_y.p')
    train_w = load_pkl_to_np(data_file_path + 'train_w.p')

    return train_x, train_y, train_w


# Load Preprocessed Data

def load_preprocessed_pd_data(data_file_path):

    train_x_pd = pd.read_pickle(data_file_path + 'train_x.p')
    train_x = np.array(train_x_pd)

    train_y_pd = pd.read_pickle(data_file_path + 'train_y.p')
    train_y = np.array(train_y_pd)

    train_w_pd = pd.read_pickle(data_file_path + 'train_w.p')
    train_w = np.array(train_w_pd)

    train_e_pd = pd.read_pickle(data_file_path + 'train_e.p')
    train_e = np.array(train_e_pd)

    test_x_pd = pd.read_pickle(data_file_path + 'test_x.p')
    test_x = np.array(test_x_pd)

    return train_x, train_y, train_w, train_e, test_x




