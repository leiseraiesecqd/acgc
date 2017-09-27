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


# LogLoss without weight
def log_loss(prob, y):

    loss = - np.sum(np.multiply(y, np.log(prob)) +
                    np.multiply((np.ones_like(y) - y), np.log(np.ones_like(prob) - prob)))

    loss /= len(y)

    return loss


# LogLoss with weight
def log_loss_with_weight(prob, y, w):

    w = w / np.sum(w)

    loss = - np.sum(np.multiply(w, (np.multiply(y, np.log(prob)) +
                                np.multiply((np.ones_like(y) - y), np.log(np.ones_like(prob) - prob)))))

    return loss


def print_loss(model, x_t, y_t, w_t, x_v, y_v, w_v):

    prob_train = model.predict(x_t)
    prob_valid = model.predict(x_v)

    loss_train = log_loss(prob_train, y_t)
    loss_valid = log_loss(prob_valid, y_v)

    loss_train_w = log_loss_with_weight(prob_train, y_t, w_t)
    loss_valid_w = log_loss_with_weight(prob_valid, y_v, w_v)

    print('Train LogLoss: {:>.8f}\n'.format(loss_train),
          'Validation LogLoss: {:>.8f}\n'.format(loss_valid),
          'Train LogLoss with Weight: {:>.8f}\n'.format(loss_train_w),
          'Validation LogLoss with Weight: {:>.8f}\n'.format(loss_valid_w))

    return loss_train, loss_valid, loss_train_w, loss_valid_w


def print_loss_proba(model, x_t, y_t, w_t, x_v, y_v, w_v):

    prob_train = np.array(model.predict_proba(x_t))[:, 1]
    prob_valid = np.array(model.predict_proba(x_v))[:, 1]

    loss_train = log_loss(prob_train, y_t)
    loss_valid = log_loss(prob_valid, y_v)

    loss_train_w = log_loss_with_weight(prob_train, y_t, w_t)
    loss_valid_w = log_loss_with_weight(prob_valid, y_v, w_v)

    print('Train LogLoss: {:>.8f}\n'.format(loss_train),
          'Validation LogLoss: {:>.8f}\n'.format(loss_valid),
          'Train LogLoss with Weight: {:>.8f}\n'.format(loss_train_w),
          'Validation LogLoss with Weight: {:>.8f}\n'.format(loss_valid_w))

    return loss_train, loss_valid, loss_train_w, loss_valid_w


# Save Grid Search Logs

def seve_grid_search_log(log_path, params, params_grid, best_score, best_parameters):

    with open(log_path + 'grid_search_log.txt', 'a') as f:
        f.write('=====================================================\n')
        f.write('Best score: {:.6f}\n'.format(best_score))
        f.wirte('Parameters:\n')
        f.wirte('\t' + str(params))
        f.wirte('Parameters Grid:\n')
        f.wirte('\t' + str(params_grid))
        f.write('Best parameters set:\n')
        for param_name in sorted(params.keys()):
            f.write('\t' + str(param_name) + ': {}\n'.format(str(best_parameters[param_name])))


# Save Final Losses

def save_loss_log(log_path, count, parameters, loss_train, loss_valid, loss_train_w, loss_valid_w):

    with open(log_path + 'loss_log.txt', 'a') as f:

        print('Saving Losses')

        f.write('-------------------- CV: {} --------------------------\n'.format(count))
        f.write('Parameters:\n')
        f.write('\t' + str(parameters))
        f.write('Losses:\n')
        f.write('\tTotal Train LogLoss: {:.6f}\n'.format(loss_train))
        f.write('\tTotal Validation LogLoss: {:.6f}\n'.format(loss_valid))
        f.write('\tTotal Train LogLoss with Weight: {:.6f}\n'.format(loss_train_w))
        f.write('\tTotal Validation LogLoss with Weight: {:.6f\n}'.format(loss_valid_w))


def save_final_loss_log(log_path, parameters, loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean):

    with open(log_path + 'loss_log.txt', 'a') as f:

        print('Saving Final Losses')

        f.write('==================== Final Losses ===================\n')
        f.write('Parameters:\n')
        f.write('\t' + str(parameters))
        f.write('Losses:\n')
        f.write('\tTotal Train LogLoss: {:.6f}\n'.format(loss_train_mean))
        f.write('\tTotal Validation LogLoss: {:.6f}\n'.format(loss_valid_mean))
        f.write('\tTotal Train LogLoss with Weight: {:.6f}\n'.format(loss_train_w_mean))
        f.write('\tTotal Validation LogLoss with Weight: {:.6f\n}'.format(loss_valid_w_mean))
        f.write('=====================================================\n')
        f.write('=====================================================\n')
