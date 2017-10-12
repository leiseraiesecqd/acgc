import pickle
import pandas as pd
import numpy as np
import os
import time
import csv
import models
import main
import preprocess
from os.path import isdir


# Save Data
def save_np_to_pkl(data, data_path):

    print('Saving ' + data_path + '...')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


# Save predictions to csv file
def save_pred_to_csv(file_path, index, prob):

    print('Saving Predictions To CSV File...')

    df = pd.DataFrame({'id': index, 'proba': prob})

    df.to_csv(file_path + 'result.csv', sep=',', index=False)


# Save probabilities of train set to csv file
def save_prob_train_to_csv(file_path, prob, label):

    print('Saving Probabilities of Train Set To CSV File...')

    df = pd.DataFrame({'prob_train': prob, 'label': label})

    df.to_csv(file_path + 'prob_train.csv', sep=',', index=True)


# Save Grid Search Logs
def seve_grid_search_log(log_path, params, params_grid, best_score, best_parameters, total_time):

    with open(log_path + 'grid_search_log.txt', 'a') as f:

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))

        f.write('=====================================================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------')
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
def save_loss_log(log_path, count, parameters, n_valid, n_cv, valid_era, loss_train,
                  loss_valid, loss_train_w, loss_valid_w, train_seed=None, cv_seed=None,
                  acc_train=None, acc_valid=None, acc_train_era=None, acc_valid_era=None):

    with open(log_path + 'loss_log.txt', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))

        f.write('===================== CV: {}/{} =====================\n'.format(count, n_cv))
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Validation Era Number: {}\n'.format(n_valid))
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
        f.write('\tCV Valid Accuracy: {:.3f}%\n'.format(acc_valid * 100))
        f.write('\tTrain Eras Accuracy:\n')
        f.write('\t\t' + str(acc_train_era) + '\n')
        f.write('\tValid Eras Accuracy:\n')
        f.write('\t\t' + str(acc_valid_era) + '\n\n')


def save_final_loss_log(log_path, parameters, n_valid, n_cv, loss_train_mean,
                        loss_valid_mean, loss_train_w_mean, loss_valid_w_mean,
                        train_seed=None, cv_seed=None, acc_train=None, acc_train_era=None):

    with open(log_path + 'loss_log.txt', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Final Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))

        f.write('==================== Final Losses ===================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Validation Era Number: {}\n'.format(n_valid))
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
        f.write('\tTotal Train Eras Accuracies:\n')
        f.write('\t\t' + str(acc_train_era) + '\n\n')
        f.write('=====================================================\n')

    with open(log_path + 'final_loss_log.txt', 'a') as f:

        print('Saving Final Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))

        f.write('=====================================================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Validation Era Number: {}\n'.format(n_valid))
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
        f.write('\tTotal Train Eras Accuracies:\n')
        f.write('\t\t' + str(acc_train_era) + '\n\n')


# Save Loss Log to csv File
def save_final_loss_log_to_csv(idx, log_path, loss_train_w_mean, loss_valid_w_mean, acc_train,
                               train_seed, cv_seed, n_valid, n_cv, parameters):

    if not os.path.isfile(log_path + 'loss_log.csv'):

        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log...')

        with open(log_path + 'loss_log.csv', 'w') as f:
            header = ['id', 'time', 'loss_train', 'loss_valid', 'train_accuracy',
                      'train_seed', 'cv_seed', 'n_valid', 'n_cv', 'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'loss_log.csv', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Final Losses to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
        log = [idx, local_time, loss_train_w_mean, loss_valid_w_mean, acc_train,
               train_seed, cv_seed, n_valid, n_cv, str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save stacking outputs of layers
def save_stack_outputs(output_path, x_outputs, test_outputs, x_g_outputs, test_g_outputs):

    print('Saving Stacking Outputs of Layer...')

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
    e_train = np.array(e_train_pd, dtype=int)

    x_test_pd = pd.read_pickle(data_file_path + 'x_test.p')
    x_test = np.array(x_test_pd, dtype=np.float64)

    id_test_pd = pd.read_pickle(data_file_path + 'id_test.p')
    id_test = np.array(id_test_pd, dtype=int)

    return x_train, y_train, w_train, e_train, x_test, id_test


# Load Preprocessed Category Data
def load_preprocessed_pd_data_g(data_file_path):

    x_g_train_pd = pd.read_pickle(data_file_path + 'x_g_train.p')
    x_g_train = np.array(x_g_train_pd, dtype=np.float64)

    x_g_test_pd = pd.read_pickle(data_file_path + 'x_g_test.p')
    x_g_test = np.array(x_g_test_pd, dtype=np.float64)

    return x_g_train, x_g_test


# Load Preprocessed Positive Data
def load_preprocessed_positive_pd_data(data_file_path):

    x_train_p_pd = pd.read_pickle(data_file_path + 'x_train_p.p')
    x_train_p = np.array(x_train_p_pd, dtype=np.float64)

    y_train_p_pd = pd.read_pickle(data_file_path + 'y_train_p.p')
    y_train_p = np.array(y_train_p_pd, dtype=np.float64)

    w_train_p_pd = pd.read_pickle(data_file_path + 'w_train_p.p')
    w_train_p = np.array(w_train_p_pd, dtype=np.float64)

    e_train_p_pd = pd.read_pickle(data_file_path + 'e_train_p.p')
    e_train_p = np.array(e_train_p_pd, dtype=int)

    x_g_train_p_pd = pd.read_pickle(data_file_path + 'x_g_train_p.p')
    x_g_train_p = np.array(x_g_train_p_pd, dtype=np.float64)

    return x_train_p, y_train_p, w_train_p, e_train_p, x_g_train_p


# Load Preprocessed Negative Data
def load_preprocessed_negative_pd_data(data_file_path):

    x_train_n_pd = pd.read_pickle(data_file_path + 'x_train_n.p')
    x_train_n = np.array(x_train_n_pd, dtype=np.float64)

    y_train_n_pd = pd.read_pickle(data_file_path + 'y_train_n.p')
    y_train_n = np.array(y_train_n_pd, dtype=np.float64)

    w_train_n_pd = pd.read_pickle(data_file_path + 'w_train_n.p')
    w_train_n = np.array(w_train_n_pd, dtype=np.float64)

    e_train_n_pd = pd.read_pickle(data_file_path + 'e_train_n.p')
    e_train_n = np.array(e_train_n_pd, dtype=int)

    x_g_train_n_pd = pd.read_pickle(data_file_path + 'x_g_train_n.p')
    x_g_train_n = np.array(x_g_train_n_pd, dtype=np.float64)

    return x_train_n, y_train_n, w_train_n, e_train_n, x_g_train_n


# Calculate LogLoss without weight
def log_loss(prob, y):

    loss = - np.sum(np.multiply(y, np.log(prob)) +
                    np.multiply((np.ones_like(y) - y), np.log(np.ones_like(prob) - prob)))

    loss /= len(y)

    return loss


# Calculate LogLoss with weight
def log_loss_with_weight(prob, y, w):

    w = w / np.sum(w)

    loss = - np.sum(np.multiply(w, (np.multiply(y, np.log(prob)) +
                                np.multiply((np.ones_like(y) - y), np.log(np.ones_like(prob) - prob)))))

    return loss


# Print Information of Gridsearch
def print_grid_info(model_name, parameters, parameters_grid):

    print('\nModel: ' + model_name + '\n')
    print("Parameters:")
    print(parameters)
    print('\n')
    print("Parameters' Grid:")
    print(parameters_grid)
    print('\n')


# Print Losses of CV of sk-learn
def print_loss_proba(model, x_t, y_t, w_t, x_v, y_v, w_v):

    prob_train = np.array(model.predict_proba(x_t))[:, 1]
    prob_valid = np.array(model.predict_proba(x_v))[:, 1]

    loss_train = log_loss(prob_train, y_t)
    loss_valid = log_loss(prob_valid, y_v)

    loss_train_w = log_loss_with_weight(prob_train, y_t, w_t)
    loss_valid_w = log_loss_with_weight(prob_valid, y_v, w_v)

    print('------------------------------------------------------')
    print('Train LogLoss: {:>.8f}\n'.format(loss_train),
          'Validation LogLoss: {:>.8f}\n'.format(loss_valid),
          'Train LogLoss with Weight: {:>.8f}\n'.format(loss_train_w),
          'Validation LogLoss with Weight: {:>.8f}'.format(loss_valid_w))

    return loss_train, loss_valid, loss_train_w, loss_valid_w


# Print Losses of LightGBM
def print_loss_lgb(model, x_t, y_t, w_t, x_v, y_v, w_v):

    prob_train = model.predict(x_t)
    prob_valid = model.predict(x_v)

    loss_train = log_loss(prob_train, y_t)
    loss_valid = log_loss(prob_valid, y_v)

    loss_train_w = log_loss_with_weight(prob_train, y_t, w_t)
    loss_valid_w = log_loss_with_weight(prob_valid, y_v, w_v)

    print('------------------------------------------------------')
    print('Train LogLoss: {:>.8f}\n'.format(loss_train),
          'Validation LogLoss: {:>.8f}\n'.format(loss_valid),
          'Train LogLoss with Weight: {:>.8f}\n'.format(loss_train_w),
          'Validation LogLoss with Weight: {:>.8f}'.format(loss_valid_w))

    return loss_train, loss_valid, loss_train_w, loss_valid_w


# Print Losses of XGBoost
def print_loss_xgb(model, x_t, y_t, w_t, x_v, y_v, w_v):

    prob_train = model.predict(models.xgb.DMatrix(x_t))
    prob_valid = model.predict(models.xgb.DMatrix(x_v))

    loss_train = log_loss(prob_train, y_t)
    loss_valid = log_loss(prob_valid, y_v)

    loss_train_w = log_loss_with_weight(prob_train, y_t, w_t)
    loss_valid_w = log_loss_with_weight(prob_valid, y_v, w_v)

    print('------------------------------------------------------')
    print('Train LogLoss: {:>.8f}\n'.format(loss_train),
          'Validation LogLoss: {:>.8f}\n'.format(loss_valid),
          'Train LogLoss with Weight: {:>.8f}\n'.format(loss_train_w),
          'Validation LogLoss with Weight: {:>.8f}\n'.format(loss_valid_w))

    return loss_train, loss_valid, loss_train_w, loss_valid_w


# Print DNN Losses
def print_loss_dnn(prob_train, prob_valid, y_t, w_t, y_v, w_v):

    loss_train = log_loss(prob_train, y_t)
    loss_valid = log_loss(prob_valid, y_v)

    loss_train_w = log_loss_with_weight(prob_train, y_t, w_t)
    loss_valid_w = log_loss_with_weight(prob_valid, y_v, w_v)

    print('------------------------------------------------------')
    print('Train LogLoss: {:>.8f}\n'.format(loss_train),
          'Validation LogLoss: {:>.8f}\n'.format(loss_valid),
          'Train LogLoss with Weight: {:>.8f}\n'.format(loss_train_w),
          'Validation LogLoss with Weight: {:>.8f}'.format(loss_valid_w))

    return loss_train, loss_valid, loss_train_w, loss_valid_w


# Print Total Losses
def print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean):

    print('------------------------------------------------------')
    print('Total Train LogLoss: {:.6f}\n'.format(loss_train_mean),
          'Total Validation LogLoss: {:.6f}\n'.format(loss_valid_mean),
          'Total Train LogLoss with Weight: {:.6f}\n'.format(loss_train_w_mean),
          'Total Validation LogLoss with Weight: {:.6f}'.format(loss_valid_w_mean))


# Print Prediction of Positive Era Rate
def print_positive_rate_test(era_sign_test=None):

    if era_sign_test is None:
        era_sign_test = load_pkl_to_np(main.prejudged_data_path + 'era_sign_test.p')

    positive_rate_test = np.sum(era_sign_test) / len(era_sign_test)

    print('------------------------------------------------------')
    print('Positive Rate Prediction of Test Set: {:.6f}%'.format(positive_rate_test * 100))


# Check if directories exit or not
def check_dir(path_list):

    for dir_path in path_list:
        if not isdir(dir_path):
            os.makedirs(dir_path)


# Check if directories exit or not
def check_dir_model(pred_path, loss_log_path=None):

    if loss_log_path is not None:
        path_list = [pred_path,
                     pred_path + 'cv_results/',
                     pred_path + 'cv_prob_train/',
                     pred_path + 'final_results/',
                     pred_path + 'final_prob_train/',
                     loss_log_path,
                     loss_log_path + 'csv_logs/']
    else:
        path_list = [pred_path,
                     pred_path + 'cv_results/',
                     pred_path + 'cv_prob_train/',]

    check_dir(path_list)


# Get Accuracy
def get_accuracy(prob, label):

    prediction = [1 if pro > 0.5 else 0 for pro in prob]
    correct_pred = [1 if p == y else 0 for p, y in zip(prediction, label)]
    accuracy = np.mean(correct_pred)

    return accuracy


# Get Accuracies of Eras
def get_era_accuracy(prob, y, e, show_accuracy):

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

        if i == len(e_sorted)-1:
            prob_era = prob_sorted[era_index]
            y_era = y_sorted[era_index]
            acc_era = get_accuracy(prob_era, y_era)
            accuracy_eras[iter_era] = acc_era
            if show_accuracy is True:
                print('Accuracy of Era-{}: {:.3f}%'.format(iter_era, acc_era * 100))
        elif ele == iter_era:
            era_index.append(i)
        else:
            prob_era = prob_sorted[era_index]
            y_era = y_sorted[era_index]
            acc_era = get_accuracy(prob_era, y_era)
            accuracy_eras[iter_era] = acc_era
            if show_accuracy is True:
                print('Accuracy of Era-{}: {:.3f}%'.format(iter_era, acc_era*100))
            iter_era = ele
            era_index = [i]

    return accuracy_eras


# Get Accuracies of Eras of  Train Probabilities
def get_train_era_accuracy(prob, y, e, show_accuracy):

    era_index = []
    accuracy_eras = {}
    iter_era = e[0]

    for i, ele in enumerate(e):

        if i == len(e)-1:
            prob_era = prob[era_index]
            y_era = y[era_index]
            acc_era = get_accuracy(prob_era, y_era)
            accuracy_eras[iter_era] = acc_era
            if show_accuracy is True:
                print('Accuracy of Era-{}: {:.3f}%'.format(iter_era, acc_era * 100))
        elif ele == iter_era:
            era_index.append(i)
        else:
            prob_era = prob[era_index]
            y_era = y[era_index]
            acc_era = get_accuracy(prob_era, y_era)
            accuracy_eras[iter_era] = acc_era
            if show_accuracy is True:
                print('Accuracy of Era-{}: {:.3f}%'.format(iter_era, acc_era*100))
            iter_era = ele
            era_index = [i]

    return accuracy_eras


# Print and Get Accuracy of Eras
def print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid, show_accuracy):

    acc_train_cv = get_accuracy(prob_train, y_train)
    acc_valid_cv = get_accuracy(prob_valid, y_valid)

    if show_accuracy is True:
        print('------------------------------------------------------')
        print('Accuracies of CV:')
        print('Accuracy of Train CV: {:.3f}%'.format(acc_train_cv * 100))
        print('Accuracy of Validation CV: {:.3f}%'.format(acc_valid_cv * 100))
        print('------------------------------------------------------')
        print('Accuracies of Train Eras:')

    acc_train_era = get_era_accuracy(prob_train, y_train, e_train, show_accuracy)

    if show_accuracy is True:
        print('------------------------------------------------------')
        print('Accuracies of Validation Eras:')

    acc_valid_era = get_era_accuracy(prob_valid, y_valid, e_valid, show_accuracy)

    return acc_train_cv, acc_valid_cv, acc_train_era, acc_valid_era


# Print and Get Accuracies of Eras of  Train Probabilities
def print_and_get_train_accuracy(prob_train, y_train, e_train, show_accuracy):

    acc_train = get_accuracy(prob_train, y_train)

    if show_accuracy is True:
        print('------------------------------------------------------')
        print('Total Train Accuracy: {:.3f}%'.format(acc_train * 100))
        print('------------------------------------------------------')
        print('Accuracies of Total Train Eras:')

    acc_train_era = get_train_era_accuracy(prob_train, y_train, e_train, show_accuracy)

    return acc_train, acc_train_era


# Check If A CV is A Bad CV
def check_bad_cv(trained_cv, valid_era):

    cv_is_trained = any(set(valid_era) == i_cv for i_cv in trained_cv)
    if cv_is_trained:
        print('This CV split has been chosen, choosing another one...')

    negative_era_counter = 0
    for era in valid_era:
        if era in preprocess.negative_era_list:
            negative_era_counter += 1
    bad_num_negative_era = negative_era_counter not in [0, 1, 2]
    if bad_num_negative_era:
        print('Bad number of negative era in this CV split, choosing another one...')

    is_bad_cv = cv_is_trained or bad_num_negative_era

    return is_bad_cv



if __name__ == '__main__':

    pass
