import pickle
import pandas as pd
import numpy as np

#111
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

    x_train = load_pkl_to_np(data_file_path + 'x_train.p')
    y_train = load_pkl_to_np(data_file_path + 'y_train.p')
    w_train = load_pkl_to_np(data_file_path + 'w_train.p')

    return x_train, y_train, w_train


# Load Preprocessed Data

def load_preprocessed_pd_data(data_file_path):

    x_train_pd = pd.read_pickle(data_file_path + 'x_train.p')
    x_train = np.array(x_train_pd)

    y_train_pd = pd.read_pickle(data_file_path + 'y_train.p')
    y_train = np.array(y_train_pd)

    w_train_pd = pd.read_pickle(data_file_path + 'w_train.p')
    w_train = np.array(w_train_pd)

    e_train_pd = pd.read_pickle(data_file_path + 'e_train.p')
    e_train = np.array(e_train_pd)

    x_test_pd = pd.read_pickle(data_file_path + 'x_test.p')
    x_test = np.array(x_test_pd)

    id_test_pd = pd.read_pickle(data_file_path + 'id_test.p')
    id_test = np.array(id_test_pd)

    return x_train, y_train, w_train, e_train, x_test, id_test


# Load Preprocessed Category Data

def load_preprocessed_pd_data_g(data_file_path):

    x_train_g_pd = pd.read_pickle(data_file_path + 'x_train_g.p')
    x_train_g = np.array(x_train_g_pd)

    x_test_g_pd = pd.read_pickle(data_file_path + 'x_test_g.p')
    x_test_g = np.array(x_test_g_pd)

    return x_train_g, x_test_g


# Save predictions to csv file

def save_pred_to_csv(file_path, id, prob):

    print('Saving predictions to csv file...')

    df = pd.DataFrame({'id': id, 'proba': prob})

    df.to_csv(file_path + 'result.csv', sep=',', index=False, float_format='%.6f')


# TODO: Save importance to csv file


# TODO: Save final loss to csv file


# TODO: LogLoss function

def log_loss(prob, y, w):

    w = w / np.sum(w)

    loss = - np.sum(np.multiply(w, (np.multiply(y, np.log(prob)) +
                                np.multiply((np.ones_like(y) - y), np.log(np.ones_like(prob) - prob)))))

    return loss



