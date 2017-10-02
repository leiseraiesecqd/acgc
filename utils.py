import pickle
import pandas as pd
import numpy as np
import os
from os.path import isdir


# Save Data
def save_np_to_pkl(data, data_path):

    print('Saving ' + data_path + '...')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


# Save predictions to csv file
def save_pred_to_csv(file_path, index, prob):

    print('Saving predictions to csv file...')

    df = pd.DataFrame({'id': index, 'proba': prob})

    df.to_csv(file_path + 'result.csv', sep=',', index=False)


# Save probabilities of train set to csv file
def save_prob_train_to_csv(file_path, prob, label):

    print('Saving probabilities of train set to csv file...')

    df = pd.DataFrame({'prob_train': prob, 'label': label})

    df.to_csv(file_path + 'prob_train.csv', sep=',', index=True)


# Save Grid Search Logs
def seve_grid_search_log(log_path, params, params_grid, best_score, best_parameters, total_time):

    with open(log_path + 'grid_search_log.txt', 'a') as f:

        f.write('=====================================================\n')
        f.write('Total Time: {:.3f}s\n'.format(total_time))
        f.write('Best Score: {:.6f}\n'.format(best_score))
        f.write('Parameters:\n')
        f.write('\t' + str(params) + '\n\n')
        f.write('Parameters Grid:\n')
        f.write('\t' + str(params_grid) + '\n\n')
        f.write('Best Parameters Set:\n')
        for param_name in sorted(params_grid.keys()):
            f.write('\t' + str(param_name) + ': {}\n'.format(str(best_parameters[param_name])))


# Save Final Losses
def save_loss_log(log_path, count, parameters, n_valid, n_cv, valid_era, loss_train, loss_valid,
                  loss_train_w, loss_valid_w, acc_train=None, acc_valid=None):

    with open(log_path + 'loss_log.txt', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Losses')

        f.write('===================== CV: {}/{} =====================\n'.format(count, n_cv))
        f.write('Validation Era: {}\n'.format(n_valid))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Validation Set Index: ' + str(valid_era) + '\n')
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tCV Train LogLoss: {:.6f}\n'.format(loss_train))
        f.write('\tCV Validation LogLoss: {:.6f}\n'.format(loss_valid))
        f.write('\tCV Train LogLoss with Weight: {:.6f}\n'.format(loss_train_w))
        f.write('\tCV Validation LogLoss with Weight: {:.6f}\n\n'.format(loss_valid_w))
        f.write('Accuracies:\n')
        f.write('\tCV Train Accuracy: {:.3f}%\n'.format(acc_train * 100))
        f.write('\tCV Valid Accuracy: {:.3f}%\n\n'.format(acc_valid * 100))


def save_final_loss_log(log_path, parameters, n_valid, n_cv, loss_train_mean, loss_valid_mean,
                        loss_train_w_mean, loss_valid_w_mean, acc_train=None):

    with open(log_path + 'loss_log.txt', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Final Losses')

        f.write('==================== Final Losses ===================\n')
        f.write('Validation Era: {}\n'.format(n_valid))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tTotal Train LogLoss: {:.6f}\n'.format(loss_train_mean))
        f.write('\tTotal Validation LogLoss: {:.6f}\n'.format(loss_valid_mean))
        f.write('\tTotal Train LogLoss with Weight: {:.6f}\n'.format(loss_train_w_mean))
        f.write('\tTotal Validation LogLoss with Weight: {:.6f}\n\n'.format(loss_valid_w_mean))
        f.write('Accuracy:\n')
        f.write('\tTotal Train Accuracy: {:.3f}%\n\n'.format(acc_train * 100))
        f.write('=====================================================\n')

    with open(log_path + 'final_loss_log.txt', 'a') as f:

        print('Saving Final Losses')

        f.write('=====================================================\n')
        f.write('Validation Era: {}\n'.format(n_valid))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tTotal Train LogLoss: {:.6f}\n'.format(loss_train_mean))
        f.write('\tTotal Validation LogLoss: {:.6f}\n'.format(loss_valid_mean))
        f.write('\tTotal Train LogLoss with Weight: {:.6f}\n'.format(loss_train_w_mean))
        f.write('\tTotal Validation LogLoss with Weight: {:.6f}\n\n'.format(loss_valid_w_mean))
        f.write('Accuracy:\n')
        f.write('\tTotal Train Accuracy: {:.3f}%\n\n'.format(acc_train * 100))


# Saving stacking outputs of layers
def save_stack_outputs(output_path, x_outputs, test_outputs, x_g_outputs, test_g_outputs):

    print('Saving stacking outputs of layer...')

    save_np_to_pkl(x_outputs, output_path + 'x_outputs.p')
    save_np_to_pkl(test_outputs, output_path + 'test_outputs.p')
    save_np_to_pkl(x_g_outputs, output_path + 'x_g_outputs.p')
    save_np_to_pkl(test_g_outputs, output_path + 'test_g_outputs.p')


# Load Data
def load_pkl_to_np(data_path):

    print('Loading ' + data_path + '...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


# Load Stacked Layer
def load_stacked_data(output_path):

    print('Loading Stacked Data...')

    x_outputs = load_pkl_to_np(output_path + 'x_outputs.p')
    test_outputs = load_pkl_to_np(output_path + 'test_outputs.p')
    x_g_outputs = load_pkl_to_np(output_path + 'x_g_outputs.p')
    test_g_outputs = load_pkl_to_np(output_path + 'test_g_outputs.p')

    return x_outputs, test_outputs, x_g_outputs, test_g_outputs


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
    x_train = np.array(x_train_pd, dtype=np.float64)

    y_train_pd = pd.read_pickle(data_file_path + 'y_train.p')
    y_train = np.array(y_train_pd, dtype=np.float64)

    w_train_pd = pd.read_pickle(data_file_path + 'w_train.p')
    w_train = np.array(w_train_pd, dtype=np.float64)

    e_train_pd = pd.read_pickle(data_file_path + 'e_train.p')
    e_train = np.array(e_train_pd, dtype=np.float64)

    x_test_pd = pd.read_pickle(data_file_path + 'x_test.p')
    x_test = np.array(x_test_pd, dtype=np.float64)

    id_test_pd = pd.read_pickle(data_file_path + 'id_test.p')
    id_test = np.array(id_test_pd, dtype=int)

    return x_train, y_train, w_train, e_train, x_test, id_test


# Load Preprocessed Category Data
def load_preprocessed_pd_data_g(data_file_path):

    x_train_g_pd = pd.read_pickle(data_file_path + 'x_train_g.p')
    x_train_g = np.array(x_train_g_pd, dtype=np.float64)

    x_test_g_pd = pd.read_pickle(data_file_path + 'x_test_g.p')
    x_test_g = np.array(x_test_g_pd, dtype=np.float64)

    return x_train_g, x_test_g


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


def print_grid_info(model_name, parameters, parameters_grid):

    print('\nModel: ' + model_name + '\n')
    print("Parameters:")
    print(parameters)
    print('\n')
    print("Parameters' grid:")
    print(parameters_grid)
    print('\n')


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


def print_loss_dnn(prob_train, prob_valid, y_t, w_t, y_v, w_v):

    loss_train = log_loss(prob_train, y_t)
    loss_valid = log_loss(prob_valid, y_v)

    loss_train_w = log_loss_with_weight(prob_train, y_t, w_t)
    loss_valid_w = log_loss_with_weight(prob_valid, y_v, w_v)

    print('Train LogLoss: {:>.8f}\n'.format(loss_train),
          'Validation LogLoss: {:>.8f}\n'.format(loss_valid),
          'Train LogLoss with Weight: {:>.8f}\n'.format(loss_train_w),
          'Validation LogLoss with Weight: {:>.8f}\n'.format(loss_valid_w))

    return loss_train, loss_valid, loss_train_w, loss_valid_w


# Check if directories exit or not
def check_dir(path_list):

    for dir_path in path_list:
        if not isdir(dir_path):
            os.makedirs(dir_path)


# Get Accuracy
def get_accuracy(prob, label):

    prediction = [1 if pro > 0.5 else 0 for pro in prob]
    correct_pred = [1 if p == y else 0 for p, y in zip(prediction, label)]
    accuracy = np.mean(correct_pred)

    return accuracy


# Print and Get Accuracy
def print_and_get_accuracy(prob_train_cv, y_train, prob_valid_cv, y_valid):

    acc_train_cv = get_accuracy(prob_train_cv, y_train)
    print('Train Accuracy: {:.3f}%'.format(acc_train_cv * 100))

    acc_valid_cv = get_accuracy(prob_valid_cv, y_valid)
    print('Valid Accuracy: {:.3f}%'.format(acc_valid_cv * 100))

    return acc_train_cv, acc_valid_cv


# Get Accuracy of Era
def get_era_accuracy(prob, y, e):

    prob_sorted = np.zeros_like(prob, dtype=np.float64)
    y_sorted = np.zeros_like(y, dtype=np.float64)
    e_sorted = np.zeros_like(e, dtype=int)

    for raw, idx in enumerate(np.argsort(e)):
        prob_sorted[raw] = prob[idx]
        y_sorted[raw] = y[idx]
        e_sorted[raw] = int(e[idx])

    era_index = []
    accuracy_eras = {}

    iter_era = e_sorted[0]

    for i, ele in enumerate(e_sorted):

        if ele == iter_era:
            era_index.append(i)
        elif i == len(e_sorted)-1:
            prob_era = prob_sorted[era_index]
            y_era = y_sorted[era_index]
            acc_era = get_accuracy(prob_era, y_era)
            accuracy_eras[iter_era] = acc_era
            print('Accuracy on Era {}: {:.3f}%'.format(iter_era, acc_era*100))
        else:
            prob_era = prob_sorted[era_index]
            y_era = y_sorted[era_index]
            acc_era = get_accuracy(prob_era, y_era)
            accuracy_eras[iter_era] = acc_era
            print('Accuracy on Era {}: {:.3f}%'.format(iter_era, acc_era*100))
            iter_era = ele
            era_index = [i]

    return accuracy_eras


if __name__ == '__main__':

    pass
