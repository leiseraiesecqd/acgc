import time
from models import utils
import os
import sys
import re
from os.path import isdir
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models.cross_validation import CrossValidation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)
color = sns.color_palette()


class ModelBase(object):
    """
        Base Model Class of Models in scikit-learn Module
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])
        self.model_name = ''
        self.num_boost_round = 0

    @staticmethod
    def get_clf(parameters):

        print('This Is Base Model!')
        clf = DecisionTreeClassifier()

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('This Is Base Model!')
        print('------------------------------------------------------')

        self.model_name = 'base'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_train, x_valid, x_test

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        # Training Model
        clf.fit(x_train, y_train, sample_weight=w_train)

        return clf

    @staticmethod
    def get_pattern():
        return None

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        # Get Classifier
        clf = self.get_clf(parameters)

        # Training Model
        if use_weight:
            clf.fit(x_train, y_train, sample_weight=w_train)
        else:
            clf.fit(x_train, y_train)

        return clf

    def fit_with_round_log(self, boost_round_log_path, cv_count, x_train, y_train, w_train,
                           x_valid, y_valid, w_valid, parameters, param_name_list, param_value_list):

        param_info = ''
        param_name = ''
        for i in range(len(param_name_list)):
            param_name += '_' + utils.get_simple_param_name(param_name_list[i])
            param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])

        boost_round_log_path += self.model_name + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + param_name + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + param_info + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += 'cv_cache/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_cv_{}_log.txt'.format(cv_count)

        print('Saving Outputs to:', boost_round_log_path)
        print('------------------------------------------------------')

        open(boost_round_log_path, 'w+').close()

        with open(boost_round_log_path, 'a') as f:
            __console__ = sys.stdout
            sys.stdout = f
            clf = self.fit(x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters)
            sys.stdout = __console__

        with open(boost_round_log_path) as f:
            lines = f.readlines()
            idx_round_cv = []
            train_loss_round_cv = []
            valid_loss_round_cv = []
            pattern = self.get_pattern()
            for line in lines:
                if pattern.match(line) is not None:
                    idx_round_cv.append(int(pattern.match(line).group(1)))
                    train_loss_round_cv.append(float(pattern.match(line).group(2)))
                    valid_loss_round_cv.append(float(pattern.match(line).group(3)))

        return clf, idx_round_cv, train_loss_round_cv, valid_loss_round_cv

    def save_boost_round_log(self, boost_round_log_path, idx_round, train_loss_round_mean, valid_loss_round_mean,
                             train_seed, cv_seed, csv_idx, parameters, param_name_list, param_value_list):

        param_info = ''
        param_name = ''
        for i in range(len(param_name_list)):
            param_name += '_' + utils.get_simple_param_name(param_name_list[i])
            param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])

        boost_round_log_path += self.model_name + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + param_name + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + param_info + '/'
        utils.check_dir([boost_round_log_path])

        utils.save_boost_round_log_to_csv(boost_round_log_path, csv_idx, idx_round, valid_loss_round_mean,
                                          train_loss_round_mean, train_seed, cv_seed, parameters)

        boost_round_log_path += 'final_logs/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_' + str(csv_idx) + '_t-' \
                                + str(train_seed) + '_c-' + str(cv_seed) + '_log.csv'

        utils.save_final_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean, valid_loss_round_mean)

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance:')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[5], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test Probability...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train Probability...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def save_csv_log(self, mode, csv_log_path, param_name_list, param_value_list,
                     csv_idx, loss_train_w_mean, loss_valid_w_mean, acc_train,
                     train_seed, cv_seed, n_valid, n_cv, parameters, file_name_params=None):

        param_info = ''
        param_name = ''
        for i in range(len(param_name_list)):
            param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])
            param_name += '_' + utils.get_simple_param_name(param_name_list[i])

        if mode == 'auto_grid_search':

            csv_log_path += self.model_name + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + param_name + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + param_info + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name

            utils.save_final_loss_log_to_csv(csv_idx, csv_log_path + param_name + '_',
                                             loss_train_w_mean, loss_valid_w_mean, acc_train,
                                             train_seed, cv_seed, n_valid, n_cv, parameters)

            csv_log_path += str(param_info) + '_'

            utils.save_final_loss_log_to_csv(csv_idx, csv_log_path, loss_train_w_mean,
                                             loss_valid_w_mean, acc_train, train_seed,
                                             cv_seed, n_valid, n_cv, parameters)

        elif mode == 'auto_train':

            csv_log_path += self.model_name + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + '_'
            if file_name_params is not None:
                for p_name in file_name_params:
                    csv_log_path += str(parameters[p_name]) + '_'
            else:
                for p_name, p_value in parameters.items():
                    csv_log_path += str(p_value) + '_'

            utils.save_final_loss_log_to_csv(csv_idx, csv_log_path, loss_train_w_mean, loss_valid_w_mean,
                                             acc_train, train_seed, cv_seed, n_valid, n_cv, parameters)
        else:

            utils.save_final_loss_log_to_csv(csv_idx, csv_log_path + self.model_name + '_',
                                             loss_train_w_mean, loss_valid_w_mean, acc_train,
                                             train_seed, cv_seed, n_valid, n_cv, parameters)

    def save_final_pred(self, mode, save_final_pred, prob_test_mean, pred_path,
                        parameters, csv_idx, train_seed, cv_seed, boost_round_log_path=None,
                        param_name_list=None, param_value_list=None, file_name_params=None):

        params = '_'
        if file_name_params is not None:
            for p_name in file_name_params:
                params += utils.get_simple_param_name(p_name) + '-' + str(parameters[p_name]) + '_'
        else:
            for p_name, p_value in parameters.items():
                params += utils.get_simple_param_name(p_name) + '-' + str(p_value) + '_'

        param_info = ''
        param_name = ''
        for i in range(len(param_name_list)):
            param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])
            param_name += '_' + utils.get_simple_param_name(param_name_list[i])

        if save_final_pred:

            if mode == 'auto_train':

                pred_path += self.model_name + '/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + params + 'results/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + str(csv_idx) + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + '_'
                utils.save_pred_to_csv(pred_path, self.id_test, prob_test_mean)

            elif mode == 'auto_train_boost_round':

                boost_round_log_path += self.model_name + '/'
                utils.check_dir([boost_round_log_path])
                boost_round_log_path += self.model_name + param_name + '/'
                utils.check_dir([boost_round_log_path])
                boost_round_log_path += self.model_name + param_info + '/'
                utils.check_dir([boost_round_log_path])
                pred_path = boost_round_log_path + 'final_results/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + str(csv_idx) + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + '_'
                utils.save_pred_to_csv(pred_path, self.id_test, prob_test_mean)

            else:
                pred_path += 'final_results/' + self.model_name + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + params
                utils.save_pred_to_csv(pred_path, self.id_test, prob_test_mean)

    @staticmethod
    def get_rescale_rate(y_train):

        positive = 0
        for y in y_train:
            if y == 1:
                positive += 1

        positive_rate = positive / len(y_train)
        rescale_rate = len(y_train) / (2*positive)

        return positive_rate, rescale_rate

    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, boost_round_log_path=None, n_valid=4,
              n_cv=20, n_era=20, train_seed=None, cv_seed=None, era_list=None, parameters=None, show_importance=False,
              show_accuracy=False, save_cv_pred=True, save_cv_prob_train=False, save_final_pred=True,
              save_final_prob_train=False, save_csv_log=True, csv_idx=None, cv_generator=None, rescale=False,
              return_prob_test=False, mode=None, param_name_list=None, param_value_list=None, file_name_params=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        if csv_idx is None:
            csv_idx = self.model_name

        # Print Start Information and Get Model Name
        self.print_start_info()

        cv_count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []
        idx_round = []
        train_loss_round_total = []
        valid_loss_round_total = []

        # Get Cross Validation Generator
        if cv_generator is None:
            cv_generator = CrossValidation.era_k_fold_with_weight

        # Training on Cross Validation Sets
        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era \
                in cv_generator(x=self.x_train, y=self.y_train, w=self.w_train, e=self.e_train,
                                n_valid=n_valid, n_cv=n_cv, n_era=n_era, seed=cv_seed, era_list=era_list):

            cv_count += 1

            # Get Positive Rate of Train Set and Rescale Rate
            positive_rate_train, rescale_rate = self.get_rescale_rate(y_train)
            positive_rate_valid, _ = self.get_rescale_rate(y_valid)

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(cv_count, n_cv))
            print('Validation Set Era: ', valid_era)
            print('Number of Features: ', x_train.shape[1])
            print('Positive Rate of Train Set: ', positive_rate_train)
            print('Positive Rate of Valid Set: ', positive_rate_valid)
            print('------------------------------------------------------')

            # Fitting and Training Model
            if mode == 'auto_train_boost_round':
                clf, idx_round_cv, train_loss_round_cv, valid_loss_round_cv = \
                    self.fit_with_round_log(boost_round_log_path, cv_count, x_train, y_train, w_train, x_valid,
                                            y_valid, w_valid, parameters, param_name_list, param_value_list)

                idx_round = idx_round_cv
                train_loss_round_total.append(train_loss_round_cv)
                valid_loss_round_total.append(valid_loss_round_cv)
            else:
                clf = self.fit(x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            # Prediction
            if save_cv_pred:
                cv_pred_path = pred_path + 'cv_results/' + self.model_name + '_cv_{}_'.format(cv_count)
            else:
                cv_pred_path = None
            prob_test = self.predict(clf, self.x_test, pred_path=cv_pred_path)

            # Save Train Probabilities to CSV File
            if save_cv_prob_train:
                cv_prob_train_path = pred_path + 'cv_prob_train/' + self.model_name + '_cv_{}_'.format(cv_count)
            else:
                cv_prob_train_path = None
            prob_train = self.get_prob_train(clf, x_train, pred_path=cv_prob_train_path)
            prob_train_all = self.get_prob_train(clf, self.x_train, pred_path=cv_prob_train_path)

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Rescale
            if rescale is True:
                print('------------------------------------------------------')
                print('Rescaling Results...')
                prob_test *= rescale_rate
                prob_train *= rescale_rate
                prob_valid *= rescale_rate

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train,
                                             prob_valid, y_valid, e_valid, show_accuracy)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + self.model_name + '_', cv_count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, train_seed, cv_seed,
                                acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era)

            prob_test_total.append(prob_test)
            prob_train_total.append(prob_train_all)
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Save Logs of num_boost_round
        if mode == 'auto_train_boost_round':
            train_loss_round_mean = np.mean(np.array(train_loss_round_total), axis=0)
            valid_loss_round_mean = np.mean(np.array(valid_loss_round_total), axis=0)
            self.save_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean, valid_loss_round_mean,
                                      train_seed, cv_seed, csv_idx, parameters, param_name_list, param_value_list)

        # Save 'num_boost_round'
        if self.model_name in ['xgb', 'lgb']:
            parameters['num_boost_round'] = self.num_boost_round

        # Save Final Result
        if save_final_pred:
            self.save_final_pred(mode, save_final_pred, prob_test_mean, pred_path,
                                 parameters, csv_idx, train_seed, cv_seed, boost_round_log_path,
                                 param_name_list, param_value_list, file_name_params=None)

        # Save Final prob_train
        if save_final_prob_train:
            utils.save_prob_train_to_csv(pred_path + 'final_prob_train/' + self.model_name + '_',
                                         prob_train_mean, self.y_train)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = \
            utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train, show_accuracy)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + self.model_name + '_', parameters, n_valid, n_cv,
                                  loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean,
                                  train_seed, cv_seed, acc_train, acc_train_era)

        # Save Loss Log to csv File
        if save_csv_log:
            self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                              loss_train_w_mean, loss_valid_w_mean, acc_train, train_seed,
                              cv_seed, n_valid, n_cv, parameters, file_name_params=file_name_params)

        # Return Final Result
        if return_prob_test:
            return prob_test_mean

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        # Select Group Variable
        x_train, x_valid, x_test = self.select_category_variable(x_train, x_g_train, x_valid,
                                                                 x_g_valid, x_test, x_g_test)

        # Print Start Information and Get Model Name
        self.print_start_info()
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Fitting and Training Model
        clf = self.fit(x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Prediction
        prob_train = self.predict(clf, x_train)
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

        # Print LogLoss
        loss_train, loss_valid, loss_train_w, loss_valid_w = \
            utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        return prob_valid, prob_test, losses

    def prejudge_train_binary(self, pred_path=None, n_splits=10, n_cv=10, cv_seed=None, use_weight=True,
                              parameters=None, show_importance=False, show_accuracy=False, cv_generator=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path)

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        # Get Cross Validation Generator
        if cv_generator is None:
            cv_generator = CrossValidation.sk_k_fold_with_weight

        # Cross Validation
        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in cv_generator(x=self.x_train, y=self.y_train, w=self.w_train,
                                                      n_splits=n_splits, n_cv=n_cv, seed=cv_seed):

            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(count, n_cv))

            # Fitting and Training Model
            clf = self.prejudge_fit(x_train, y_train, w_train, x_valid, y_valid, w_valid,
                                    parameters=parameters, use_weight=use_weight)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path +
                                     'cv_results/' + self.model_name + '_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, x_train)
            prob_train_all = self.get_prob_train(clf, self.x_train)

            # Prediction
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(prob_train, self.y_train, self.w_train, prob_valid, y_valid, w_valid)

            prob_test_total.append(prob_test)
            prob_train_total.append(prob_train_all)
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        _, _ = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train, show_accuracy)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/' + self.model_name + '_', self.id_test, prob_test_mean)

        return prob_test_mean

    def prejudge_stack_train(self, x_train, x_g_train, y_train, w_train, e_train, x_valid,
                             x_g_valid, y_valid, w_valid, e_valid, x_test, x_g_test, id_test,
                             pred_path=None, loss_log_path=None, csv_log_path=None, n_valid=4, n_cv=20,
                             train_seed=None, cv_seed=None, parameters=None, show_importance=False,
                             show_accuracy=False, save_final_pred=True, save_final_prob_train=False,
                             save_csv_log=True, csv_idx=None, mode=None, file_name_params=None,
                             param_name_list=None, param_value_list=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        if csv_idx is None:
            csv_idx = self.model_name

        # Print Start Information and Get Model Name
        self.print_start_info()

        # Select Group Variable
        x_train, x_valid, x_test = self.select_category_variable(x_train, x_g_train, x_valid,
                                                                 x_g_valid, x_test, x_g_test)

        print('======================================================')
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Fitting and Training Model
        clf = self.fit(x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Test Probabilities
        prob_test = self.predict(clf, x_test)

        # Train Probabilities
        prob_train = self.get_prob_train(clf, x_train)

        # Get Probabilities of Validation Set
        prob_valid = self.predict(clf, x_valid)

        # Print LogLoss
        loss_train, loss_valid, loss_train_w, loss_valid_w = \
            utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

        # Save 'num_boost_round'
        if self.model_name in ['xgb', 'lgb']:
            parameters['num_boost_round'] = self.num_boost_round

        # Save Final Result
        if save_final_pred:
            self.save_final_pred(mode, save_final_pred, prob_test, pred_path,
                                 parameters, csv_idx, train_seed, cv_seed, file_name_params=file_name_params)

        # Print Total Losses
        utils.print_total_loss(loss_train, loss_valid, loss_train_w, loss_valid_w)
        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Print and Get Accuracies of CV
        acc_train, acc_valid, acc_train_era, acc_valid_era = \
            utils.print_and_get_accuracy(prob_train, y_train, e_train,
                                         prob_valid, y_valid, e_valid, show_accuracy)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + self.model_name + '_', parameters, n_valid, n_cv,
                                  loss_train, loss_valid, loss_train_w, loss_valid_w,
                                  train_seed, cv_seed, acc_train, acc_train_era)

        # Save Loss Log to csv File
        if save_csv_log:
            self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                              loss_train_w, loss_valid_w, acc_train, train_seed,
                              cv_seed, n_valid, n_cv, parameters, file_name_params=file_name_params)

        return prob_valid, prob_test, losses


class LRegression(ModelBase):
    """
        Logistic Regression
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = LogisticRegression(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Logistic Regression...')
        print('------------------------------------------------------')

        self.model_name = 'lr'

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')
        self.importance = np.abs(clf.coef_)[0]
        indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, indices[f], self.importance[indices[f]]))


class KNearestNeighbor(ModelBase):
    """
        k-Nearest Neighbor Classifier
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = KNeighborsClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training k-Nearest Neighbor Classifier...')
        print('------------------------------------------------------')

        self.model_name = 'knn'


class SupportVectorClustering(ModelBase):
    """
        SVM - Support Vector Clustering
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = SVC(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Support Vector Clustering...')
        print('------------------------------------------------------')

        self.model_name = 'svc'


class Gaussian(ModelBase):
    """
        Gaussian NB
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = GaussianNB(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Gaussian...')
        print('------------------------------------------------------')

        self.model_name = 'gs'


class DecisionTree(ModelBase):
    """
        Decision Tree
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = DecisionTreeClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Decision Tree...')
        print('------------------------------------------------------')

        self.model_name = 'dt'


class RandomForest(ModelBase):
    """
        Random Forest
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = RandomForestClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Random Forest...')
        print('------------------------------------------------------')

        self.model_name = 'rf'


class ExtraTrees(ModelBase):
    """
        Extra Trees
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = ExtraTreesClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Extra Trees...')
        print('------------------------------------------------------')

        self.model_name = 'et'


class AdaBoost(ModelBase):
    """
        AdaBoost
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = AdaBoostClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training AdaBoost...')
        print('------------------------------------------------------')

        self.model_name = 'ab'


class GradientBoosting(ModelBase):
    """
        Gradient Boosting
    """
    @staticmethod
    def get_clf(parameters):

        print('Initialize Model...')
        clf = GradientBoostingClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Gradient Boosting...')
        print('------------------------------------------------------')

        self.model_name = 'gb'


class XGBoost(ModelBase):
    """
        XGBoost
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, num_boost_round):

        super(XGBoost, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, id_te)

        self.num_boost_round = num_boost_round

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training XGBoost...')
        print('------------------------------------------------------')

        self.model_name = 'xgb'

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)

        # Booster
        eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]
        bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

        return bst

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        if use_weight:
            d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)
        else:
            d_train = xgb.DMatrix(x_train, label=y_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid)

        # Booster
        eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]
        bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

        return bst

    @staticmethod
    def get_pattern():
        return re.compile(r'\[(\d*)\]\tTrain-logloss:(.*)\tValid-logloss:(.*)')

    def get_importance(self, model):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = model.get_fscore()
        sorted_importance = sorted(self.importance.items(), key=lambda d: d[1], reverse=True)

        feature_num = len(self.importance)

        for i in range(feature_num):
            print('{} | feature {} | {}'.format(i + 1, sorted_importance[i][0], sorted_importance[i][1]))

    def predict(self, model, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test Probability...')

        prob_test = model.predict(xgb.DMatrix(x_test))

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, model, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train Probability...')

        prob_train = model.predict(xgb.DMatrix(x_train))

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train


class SKLearnXGBoost(ModelBase):
    """
        XGBoost using sklearn module
    """
    @staticmethod
    def get_clf(parameters=None):

        print('Initialize Model...')
        clf = XGBClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training XGBoost(sklearn)...')
        print('------------------------------------------------------')

        self.model_name = 'xgb_sk'

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        # Training Model
        clf.fit(x_train, y_train, sample_weight=w_train,
                eval_set=[(x_train, y_train), (x_valid, y_valid)],
                early_stopping_rounds=100, eval_metric='logloss', verbose=True)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, self.indices[f], self.importance[self.indices[f]]))


class LightGBM(ModelBase):
    """
        LightGBM
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, num_boost_round):

        super(LightGBM, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, id_te)

        self.num_boost_round = num_boost_round

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training LightGBM...')
        print('------------------------------------------------------')

        self.model_name = 'lgb'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_g_train, x_g_valid, x_g_test

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Create Dataset
        idx_category = [x_train.shape[1] - 1]
        # print('Index of categorical feature: {}'.format(idx_category))
        # print('------------------------------------------------------')

        d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
        d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)

        # Booster
        bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                        valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

        return bst

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        if use_weight:
            d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
            d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)
        else:
            d_train = lgb.Dataset(x_train, label=y_train, categorical_feature=idx_category)
            d_valid = lgb.Dataset(x_valid, label=y_valid, categorical_feature=idx_category)

        # Booster
        bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                        valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

        return bst

    @staticmethod
    def get_pattern():
        return re.compile(r"\[(\d*)\]\tTrain\'s binary_logloss: (.*)\tValid\'s binary_logloss:(.*)")

    @staticmethod
    def logloss_obj(y, pred):

        grad = (pred - y) / ((1 - pred) * pred)
        hess = (pred * pred - 2 * pred * y + y) / ((1 - pred) * (1 - pred) * pred * pred)

        return grad, hess

    def get_importance(self, bst):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = bst.feature_importance()
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

        print('\n')

    def predict(self, bst, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test Probability...')

        prob_test = bst.predict(x_test)

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, bst, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train Probability...')

        prob_train = bst.predict(x_train)

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def prejudge_train_multiclass(self, pred_path=None, n_splits=10, n_cv=10, n_era=20, cv_seed=None,
                                  use_weight=True, parameters=None, show_importance=False, cv_generator=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path)

        cv_counter = 0
        prob_test_total = np.array([])

        # Get Cross Validation Generator
        if cv_generator is None:
            cv_generator = CrossValidation.sk_k_fold_with_weight

        # Cross Validation
        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in cv_generator(x=self.x_train, y=self.y_train, w=self.w_train,
                                                      n_splits=n_splits, n_cv=n_cv, seed=cv_seed):

            cv_counter += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(cv_counter, n_cv))

            idx_category = [x_train.shape[1] - 1]
            print('Index of categorical feature: {}'.format(idx_category))
            print('------------------------------------------------------')

            if use_weight:
                d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
                d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)
            else:
                d_train = lgb.Dataset(x_train, label=y_train, categorical_feature=idx_category)
                d_valid = lgb.Dataset(x_valid, label=y_valid, categorical_feature=idx_category)

            # Booster
            bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                            valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

            # Feature Importance
            if show_importance:
                self.get_importance(bst)

            # Prediction
            prob_test = self.predict(bst, self.x_test)

            if cv_counter == 1:
                prob_test_total = prob_test.reshape(-1, 1, n_era)
            else:
                np.concatenate((prob_test_total, prob_test.reshape(-1, 1, n_era)), axis=1)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(prob_test_total, axis=1)

        return prob_test_mean


class SKLearnLightGBM(ModelBase):
    """
        LightGBM using sklearn module
    """
    @staticmethod
    def get_clf(parameters=None):

        print('Initialize Model...')
        clf = LGBMClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training LightGBM(sklearn)...')
        print('------------------------------------------------------')

        self.model_name = 'lgb_sk'

    @staticmethod
    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))

        # Fitting and Training Model
        clf.fit(x_train, y_train, sample_weight=w_train, categorical_feature=idx_category,
                eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_names=['train', 'eval'],
                early_stopping_rounds=100, eval_sample_weight=[w_train, w_valid],
                eval_metric='logloss', verbose=True)

        return clf


class CatBoost(ModelBase):
    """
        CatBoost
    """
    @staticmethod
    def get_clf(parameters=None):

        clf = CatBoostClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training CatBoost...')
        print('------------------------------------------------------')

        self.model_name = 'cb'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_g_train, x_g_valid, x_g_test

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        idx_category = [x_train.shape[1] - 1]
        # print('Index of categorical feature: {}'.format(idx_category))
        # print('------------------------------------------------------')

        # Convert Zeros in Weights to Small Positive Numbers
        w_train = [0.001 if w == 0 else w for w in w_train]

        # Fitting and Training Model
        clf.fit(X=x_train, y=y_train, cat_features=idx_category, sample_weight=w_train,
                baseline=None, use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)

        return clf

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        # Get Classifier
        clf = self.get_clf(parameters)

        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        # Convert Zeros in Weights to Small Positive Numbers
        w_train = [0.001 if w == 0 else w for w in w_train]

        # Fitting and Training Model
        if use_weight:
            clf.fit(X=x_train, y=y_train, cat_features=idx_category, sample_weight=w_train,
                    baseline=None, use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)
        else:
            clf.fit(X=x_train, y=y_train, cat_features=idx_category, baseline=None,
                    use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)

        return clf

    @staticmethod
    def get_pattern():

        # 0:	learn 0.6930110854	test 0.6932013607	bestTest 0.6932013607		total: 1.11s	remaining: 1m 36s
        return re.compile(r'(\d*):\tlearn (.*)\ttest (.*)\tbestTest')

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))


class DeepNeuralNetworks(ModelBase):
    """
        Deep Neural Networks
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, parameters):

        super(DeepNeuralNetworks, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, id_te)

        # Hyperparameters
        self.parameters = parameters
        self.version = parameters['version']
        self.epochs = parameters['epochs']
        self.unit_number = parameters['unit_number']
        self.learning_rate = parameters['learning_rate']
        self.keep_probability = parameters['keep_probability']
        self.batch_size = parameters['batch_size']
        self.train_seed = parameters['seed']
        self.display_step = parameters['display_step']
        self.save_path = parameters['save_path']
        self.log_path = parameters['log_path']

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Deep Neural Networks...')
        print('------------------------------------------------------')

        self.model_name = 'dnn'

    @staticmethod
    def get_pattern():

        # [0] | CV: 10 | Epoch: 4/4 | Batch: 7300 | Time: 352.85s | Train_Loss: 0.71237314 | Valid_Loss: 0.72578128
        return re.compile(r'\[(\d*)\].*Train_Loss: (.*) \| Valid_Loss: (.*)')

    # Input Tensors
    def input_tensor(self):

        feature_num = self.x_train.shape[1]

        inputs_ = tf.placeholder(tf.float32, [None, feature_num], name='inputs')
        labels_ = tf.placeholder(tf.float32, None, name='labels')
        loss_weights_ = tf.placeholder(tf.float32, None, name='loss_weights')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
        is_training_ = tf.placeholder(tf.bool, name='is_training')

        return inputs_, labels_, loss_weights_, learning_rate_, keep_prob_, is_training_

    # Full Connected Layer
    def fc_layer(self, x_tensor, layer_name, num_outputs, keep_prob, is_training):

        if is_training is True:
            print('Using Batch Normalization')

        with tf.name_scope(layer_name):

            # x_shape = features.get_shape().as_list()
            # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
            # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
            #                                                                      seed=self.train_seed)
            weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)
            weights_reg = tf.contrib.layers.l2_regularizer(1e-3)
            normalizer_fn = tf.contrib.layers.batch_norm
            normalizer_params = {'is_training': is_training}

            fc = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                   num_outputs=num_outputs,
                                                   activation_fn=tf.nn.sigmoid,
                                                   weights_initializer=weights_initializer,
                                                   weights_regularizer=weights_reg,
                                                   normalizer_fn=normalizer_fn,
                                                   normalizer_params=normalizer_params,
                                                   biases_initializer=tf.zeros_initializer(dtype=tf.float32))

            tf.summary.histogram('fc_layer', fc)

            fc = tf.nn.dropout(fc, keep_prob)

        return fc

    # Output Layer
    def output_layer(self, x_tensor, layer_name, num_outputs):

        with tf.name_scope(layer_name):

            # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
            # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
            #                                                                      seed=self.train_seed)
            weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)

            out = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                    num_outputs=num_outputs,
                                                    activation_fn=None,
                                                    weights_initializer=weights_initializer,
                                                    biases_initializer=tf.zeros_initializer(dtype=tf.float32))

        return out

    # Model
    def model(self, x, n_unit, keep_prob, is_training):

        #  fc1 = fc_layer(x, 'fc1', n_unit[0], keep_prob)
        #  fc2 = fc_layer(fc1, 'fc2', n_unit[1], keep_prob)
        #  fc3 = fc_layer(fc2, 'fc3', n_unit[2], keep_prob)
        #  fc4 = fc_layer(fc3, 'fc4', n_unit[3], keep_prob)
        #  fc5 = fc_layer(fc4, 'fc5', n_unit[4], keep_prob)
        #  logit_ = self.output_layer(fc5, 'output', 1)

        fc = [x]

        for i in range(len(n_unit)):
            fc.append(self.fc_layer(fc[i], 'fc{}'.format(i + 1), n_unit[i], keep_prob, is_training))

        logit_ = self.output_layer(fc[len(n_unit)], 'output', 1)

        return logit_

    # LogLoss
    @staticmethod
    def log_loss(logit, w, y):

        with tf.name_scope('prob'):
            logit = tf.squeeze(logit)
            prob = tf.nn.sigmoid(logit)

        with tf.name_scope('log_loss'):

            w = w / tf.reduce_sum(w)
            ones = tf.ones_like(y, dtype=tf.float32)
            loss = - tf.reduce_sum(w * (y * tf.log(prob) + (ones-y) * tf.log(ones-prob)))
            # loss = tf.losses.log_loss(labels=y, predictions=prob, weights=w)

        tf.summary.scalar('log_loss', loss)

        return loss

    # Get Batches
    @staticmethod
    def get_batches(x, y, w, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num+1, batch_num):

            if ii != n_batches * batch_num - 1:
                batch_x, batch_y, batch_w = x[ii: ii + batch_num], y[ii: ii + batch_num], w[ii: ii + batch_num]
            else:
                batch_x, batch_y, batch_w = x[ii:], y[ii:], w[ii:]

            yield batch_x, batch_y, batch_w

    # Get Batches for Prediction
    @staticmethod
    def get_batches_for_predict(x, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num + 1, batch_num):

            if ii != n_batches * batch_num - 1:
                batch_x = x[ii: ii + batch_num]
            else:
                batch_x = x[ii:]

            yield batch_x

    # Get Probabilities
    def get_prob(self, sess, logits, x, batch_num, inputs, keep_prob, is_training):

        logits_pred = np.array([])

        for x_batch in self.get_batches_for_predict(x, batch_num):

            logits_pred_batch = sess.run(logits, {inputs: x_batch, keep_prob: 1.0, is_training: False})
            logits_pred_batch = logits_pred_batch.flatten()
            logits_pred = np.concatenate((logits_pred, logits_pred_batch))

        prob = 1.0 / (1.0 + np.exp(-logits_pred))

        return prob

    # Trainer
    def trainer(self, sess, cv_counter, x_train, y_train, w_train, x_valid, y_valid, w_valid,
                optimizer, merged, cost_, inputs, labels, weights, lr, keep_prob, is_training, start_time):

        train_log_path = self.log_path + self.version + '/cv_{}/train'.format(cv_counter)
        valid_log_path = self.log_path + self.version + '/cv_{}/valid'.format(cv_counter)

        if not isdir(train_log_path):
            os.makedirs(train_log_path)
        if not isdir(valid_log_path):
            os.makedirs(valid_log_path)

        train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        sess.run(tf.global_variables_initializer())

        batch_counter = 0
        idx = 0

        for epoch_i in range(self.epochs):

            for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                   y_train,
                                                                                   w_train,
                                                                                   self.batch_size)):

                batch_counter += 1

                _, cost = sess.run([optimizer, cost_],
                                   {inputs: batch_x,
                                    labels: batch_y,
                                    weights: batch_w,
                                    lr: self.learning_rate,
                                    keep_prob: self.keep_probability,
                                    is_training: True})

                if str(cost) == 'nan':
                    raise ValueError('NaN BUG!!! Try Another Seed!!!')

                if batch_counter % self.display_step == 0 and batch_i > 0:

                    idx += 1

                    summary_train, cost_train = sess.run([merged, cost_],
                                                         {inputs: batch_x,
                                                          labels: batch_y,
                                                          weights: batch_w,
                                                          keep_prob: 1.0,
                                                          is_training: False})
                    train_writer.add_summary(summary_train, batch_counter)

                    cost_valid_all = []

                    for iii, (valid_batch_x,
                              valid_batch_y,
                              valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                           y_valid,
                                                                           w_valid,
                                                                           self.batch_size)):
                        summary_valid_i, cost_valid_i = sess.run([merged, cost_],
                                                                 {inputs: valid_batch_x,
                                                                  labels: valid_batch_y,
                                                                  weights: valid_batch_w,
                                                                  keep_prob: 1.0,
                                                                  is_training: False})

                        valid_writer.add_summary(summary_valid_i, batch_counter)

                        cost_valid_all.append(cost_valid_i)

                    cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                    total_time = time.time() - start_time

                    print('[{}] |'.format(idx),
                          'CV: {} |'.format(cv_counter),
                          'Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                          'Batch: {} |'.format(batch_counter),
                          'Time: {:3.2f}s |'.format(total_time),
                          'Train_Loss: {:.8f} |'.format(cost_train),
                          'Valid_Loss: {:.8f}'.format(cost_valid))

    def train_with_round_log(self, boost_round_log_path, sess, cv_counter, x_train, y_train, w_train,
                             x_valid, y_valid, w_valid, optimizer, merged, cost_, inputs, labels, weights,
                             lr, keep_prob, is_training, start_time, param_name_list, param_value_list):

        param_info = ''
        param_name = ''
        for i in range(len(param_name_list)):
            param_name += '_' + utils.get_simple_param_name(param_name_list[i])
            param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])

        boost_round_log_path += self.model_name + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + param_name + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + param_info + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += 'cv_cache/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_cv_{}_log.txt'.format(cv_counter)

        print('Saving Outputs to:', boost_round_log_path)
        print('------------------------------------------------------')

        open(boost_round_log_path, 'w+').close()

        with open(boost_round_log_path, 'a') as f:
            __console__ = sys.stdout
            sys.stdout = f
            self.trainer(sess, cv_counter, x_train, y_train, w_train, x_valid, y_valid, w_valid,
                         optimizer, merged, cost_, inputs, labels, weights, lr, keep_prob, is_training, start_time)
            sys.stdout = __console__

        with open(boost_round_log_path) as f:
            lines = f.readlines()
            idx_round_cv = []
            train_loss_round_cv = []
            valid_loss_round_cv = []
            pattern = self.get_pattern()
            for line in lines:
                if pattern.match(line) is not None:
                    idx_round_cv.append(int(pattern.match(line).group(1)))
                    train_loss_round_cv.append(float(pattern.match(line).group(2)))
                    valid_loss_round_cv.append(float(pattern.match(line).group(3)))

        return idx_round_cv, train_loss_round_cv, valid_loss_round_cv

    # Training
    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, boost_round_log_path=None, n_valid=4,
              n_cv=20, n_era=20, train_seed=None, cv_seed=None, era_list=None, parameters=None, show_importance=False,
              show_accuracy=False, save_cv_pred=True, save_cv_prob_train=False, save_final_pred=True,
              save_final_prob_train=False, save_csv_log=True, csv_idx=None, cv_generator=None, rescale=False,
              return_prob_test=False, mode=None, param_name_list=None, param_value_list=None, file_name_params=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        if csv_idx is None:
            csv_idx = self.model_name

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_training = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_training)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        # Training
        self.print_start_info()

        with tf.Session(graph=train_graph) as sess:

            # Merge all the summaries
            merged = tf.summary.merge_all()

            start_time = time.time()
            cv_counter = 0

            prob_test_total = []
            prob_train_total = []
            loss_train_total = []
            loss_valid_total = []
            loss_train_w_total = []
            loss_valid_w_total = []
            idx_round = []
            train_loss_round_total = []
            valid_loss_round_total = []

            # Get Cross Validation Generator
            if cv_generator is None:
                cv_generator = CrossValidation.era_k_fold_with_weight

            for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era \
                    in cv_generator(self.x_train, self.y_train, self.w_train, self.e_train,
                                    n_valid=n_valid, n_cv=n_cv, n_era=n_era, seed=cv_seed, era_list=era_list):

                cv_counter += 1

                # Get Positive Rate of Train Set and Rescale Rate
                positive_rate_train, rescale_rate = self.get_rescale_rate(y_train)
                positive_rate_valid, _ = self.get_rescale_rate(y_valid)

                print('======================================================')
                print('Training on the Cross Validation Set: {}/{}'.format(cv_counter, n_cv))
                print('Number of Features: ', x_train.shape[1])
                print('Validation Set Era: ', valid_era)
                print('Positive Rate of Train Set: ', positive_rate_train)
                print('Positive Rate of Valid Set: ', positive_rate_valid)
                print('Rescale Rate of Valid Set: ', rescale_rate)
                print('------------------------------------------------------')

                # Training
                if mode == 'auto_train_boost_round':
                    idx_round_cv, train_loss_round_cv, valid_loss_round_cv = \
                        self.train_with_round_log(boost_round_log_path, sess, cv_counter, x_train, y_train,
                                                  w_train, x_valid, y_valid, w_valid, optimizer, merged, cost_,
                                                  inputs, labels, weights, lr, keep_prob, is_training,
                                                  start_time, param_name_list, param_value_list)
                    idx_round = idx_round_cv
                    train_loss_round_total.append(train_loss_round_cv)
                    valid_loss_round_total.append(valid_loss_round_cv)
                else:
                    self.trainer(sess, cv_counter, x_train, y_train, w_train, x_valid, y_valid, w_valid, optimizer,
                                 merged, cost_, inputs, labels, weights, lr, keep_prob, is_training, start_time)

                # Save Model
                # print('Saving model...')
                # saver = tf.train.Saver()
                # saver.save(sess, self.save_path + 'model.' + self.version + '.ckpt')

                # Prediction
                print('------------------------------------------------------')
                print('Predicting Probabilities...')
                prob_train = self.get_prob(sess, logits, x_train, self.batch_size, inputs, keep_prob, is_training)
                prob_train_all = self.get_prob(sess, logits, self.x_train, self.batch_size, inputs, keep_prob, is_training)
                prob_valid = self.get_prob(sess, logits, x_valid, self.batch_size, inputs, keep_prob, is_training)
                prob_test = self.get_prob(sess, logits, self.x_test, self.batch_size, inputs, keep_prob, is_training)

                # Rescale
                if rescale is True:
                    print('------------------------------------------------------')
                    print('Rescaling Results...')
                    prob_test *= rescale_rate
                    prob_train *= rescale_rate
                    prob_valid *= rescale_rate
                
                # Print Losses of CV
                loss_train, loss_valid, loss_train_w, loss_valid_w = \
                    utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

                prob_test_total.append(prob_test)
                prob_train_total.append(prob_train_all)
                loss_train_total.append(loss_train)
                loss_valid_total.append(loss_valid)
                loss_train_w_total.append(loss_train_w)
                loss_valid_w_total.append(loss_valid_w)

                # Print and Get Accuracies of CV
                acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                    utils.print_and_get_accuracy(prob_train, y_train, e_train,
                                                 prob_valid, y_valid, e_valid, show_accuracy)

                utils.save_loss_log(loss_log_path + self.model_name + '_', cv_counter, self.parameters, n_valid, n_cv,
                                    valid_era, loss_train, loss_valid, loss_train_w, loss_valid_w, train_seed, cv_seed,
                                    acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era)

                if save_cv_pred:
                    utils.save_pred_to_csv(pred_path + 'cv_results/' + self.model_name + '_cv_{}_'.format(cv_counter),
                                           self.id_test, prob_test)

            # Final Result
            print('======================================================')
            print('Calculating Final Result...')

            prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
            prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
            loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
            loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
            loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
            loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

            # Save Logs of num_boost_round
            if mode == 'auto_train_boost_round':
                l = len(train_loss_round_total[0])
                for train_loss_cv in train_loss_round_total:
                    if l > len(train_loss_cv):
                        l = len(train_loss_cv)
                idx_round = idx_round[:l]
                train_loss_round_total = [train_loss[:l] for train_loss in train_loss_round_total]
                valid_loss_round_total = [valid_loss[:l] for valid_loss in valid_loss_round_total]
                train_loss_round_mean = np.mean(np.array(train_loss_round_total), axis=0)
                valid_loss_round_mean = np.mean(np.array(valid_loss_round_total), axis=0)
                self.save_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean, valid_loss_round_mean,
                                          train_seed, cv_seed, csv_idx, parameters, param_name_list, param_value_list)

            # Save Final Result
            if save_final_pred:
                self.save_final_pred(mode, save_final_pred, prob_test_mean, pred_path,
                                     parameters, csv_idx, train_seed, cv_seed, boost_round_log_path,
                                     param_name_list, param_value_list, file_name_params=None)

            # Save Final prob_train
            if save_final_prob_train:
                utils.save_prob_train_to_csv(pred_path + 'final_prob_train/' + self.model_name + '_',
                                             prob_train_mean, self.y_train)

            # Print Total Losses
            utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

            # Print and Get Accuracies of CV of All Train Set
            acc_train, acc_train_era = \
                utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train, show_accuracy)

            # Save Final Losses to File
            utils.save_final_loss_log(loss_log_path + self.model_name + '_', self.parameters, n_valid, n_cv,
                                      loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean,
                                      train_seed, cv_seed, acc_train, acc_train_era)

            # Save Loss Log to csv File
            if save_csv_log:
                self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                                  loss_train_w_mean, loss_valid_w_mean, acc_train, train_seed,
                                  cv_seed, n_valid, n_cv, parameters, file_name_params=file_name_params)

            # Return Final Result
            if return_prob_test:
                return prob_test_mean

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters=None, show_importance=False):

        # Print Start Information
        self.print_start_info()
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_train = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        with tf.Session(graph=train_graph) as sess:

            start_time = time.time()
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(self.epochs):

                batch_counter = 0

                for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                       y_train,
                                                                                       w_train,
                                                                                       self.batch_size)):

                    batch_counter += 1

                    _, cost_train = sess.run([optimizer, cost_],
                                             {inputs: batch_x,
                                              labels: batch_y,
                                              weights: batch_w,
                                              lr: self.learning_rate,
                                              keep_prob: self.keep_probability,
                                              is_train: True})

                    if str(cost_train) == 'nan':
                        raise ValueError('NaN BUG!!! Try Another Seed!!!')

                    if batch_counter % self.display_step == 0 and batch_i > 0:

                        cost_valid_all = []

                        for iii, (valid_batch_x,
                                  valid_batch_y,
                                  valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                               y_valid,
                                                                               w_valid,
                                                                               self.batch_size)):
                            cost_valid_i = sess.run(cost_, {inputs: valid_batch_x,
                                                            labels: valid_batch_y,
                                                            weights: valid_batch_w,
                                                            keep_prob: 1.0,
                                                            is_train: False})

                            cost_valid_all.append(cost_valid_i)

                        cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                        total_time = time.time() - start_time

                        print('Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                              'Batch: {} |'.format(batch_counter),
                              'Time: {:>3.2f}s |'.format(total_time),
                              'Train_Loss: {:>.8f} |'.format(cost_train),
                              'Valid_Loss: {:>.8f}'.format(cost_valid))

            # Prediction
            print('------------------------------------------------------')
            print('Predicting Probabilities...')

            logits_pred_train = sess.run(logits, {inputs: x_train, keep_prob: 1.0, is_train: False})
            logits_pred_valid = sess.run(logits, {inputs: x_valid, keep_prob: 1.0, is_train: False})
            logits_pred_test = sess.run(logits, {inputs: x_test, keep_prob: 1.0, is_train: False})

            logits_pred_train = logits_pred_train.flatten()
            logits_pred_valid = logits_pred_valid.flatten()
            logits_pred_test = logits_pred_test.flatten()

            prob_train = 1.0 / (1.0 + np.exp(-logits_pred_train))
            prob_valid = 1.0 / (1.0 + np.exp(-logits_pred_valid))
            prob_test = 1.0 / (1.0 + np.exp(-logits_pred_test))

            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

            losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

            return prob_valid, prob_test, losses

    def prejudge_stack_train(self, x_train, x_g_train, y_train, w_train, e_train, x_valid,
                             x_g_valid, y_valid, w_valid, e_valid, x_test, x_g_test, id_test,
                             pred_path=None, loss_log_path=None, csv_log_path=None, n_valid=4, n_cv=20,
                             train_seed=None, cv_seed=None, parameters=None, show_importance=False,
                             show_accuracy=False, save_final_pred=True, save_final_prob_train=False,
                             save_csv_log=True, csv_idx=None, return_prob_test=False,
                             mode=None, file_name_params=None, param_name_list=None, param_value_list=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        if csv_idx is None:
            csv_idx = self.model_name

        # Print Start Information and Get Model Name
        self.print_start_info()

        print('======================================================')
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_train = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        with tf.Session(graph=train_graph) as sess:

            start_time = time.time()
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(self.epochs):

                batch_counter = 0

                for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                       y_train,
                                                                                       w_train,
                                                                                       self.batch_size)):

                    batch_counter += 1

                    _, cost_train = sess.run([optimizer, cost_],
                                             {inputs: batch_x,
                                              labels: batch_y,
                                              weights: batch_w,
                                              lr: self.learning_rate,
                                              keep_prob: self.keep_probability,
                                              is_train: True})

                    if str(cost_train) == 'nan':
                        raise ValueError('NaN BUG!!! Try Another Seed!!!')

                    if batch_counter % self.display_step == 0 and batch_i > 0:

                        cost_valid_all = []

                        for iii, (valid_batch_x,
                                  valid_batch_y,
                                  valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                               y_valid,
                                                                               w_valid,
                                                                               self.batch_size)):
                            cost_valid_i = sess.run(cost_, {inputs: valid_batch_x,
                                                            labels: valid_batch_y,
                                                            weights: valid_batch_w,
                                                            keep_prob: 1.0,
                                                            is_train: False})

                            cost_valid_all.append(cost_valid_i)

                        cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                        total_time = time.time() - start_time

                        print('Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                              'Batch: {} |'.format(batch_counter),
                              'Time: {:>3.2f}s |'.format(total_time),
                              'Train_Loss: {:>.8f} |'.format(cost_train),
                              'Valid_Loss: {:>.8f}'.format(cost_valid))

            # Prediction
            print('------------------------------------------------------')
            print('Predicting Train Probabilities...')

            logits_pred_train = sess.run(logits, {inputs: x_train, keep_prob: 1.0, is_train: False})
            logits_pred_valid = sess.run(logits, {inputs: x_valid, keep_prob: 1.0, is_train: False})
            logits_pred_test = sess.run(logits, {inputs: x_test, keep_prob: 1.0, is_train: False})

            logits_pred_train = logits_pred_train.flatten()
            logits_pred_valid = logits_pred_valid.flatten()
            logits_pred_test = logits_pred_test.flatten()

            prob_train = 1.0 / (1.0 + np.exp(-logits_pred_train))
            prob_valid = 1.0 / (1.0 + np.exp(-logits_pred_valid))
            prob_test = 1.0 / (1.0 + np.exp(-logits_pred_test))

            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

            # Save Final Result
            if save_final_pred:
                self.save_final_pred(mode, save_final_pred, prob_test, pred_path,
                                     parameters, csv_idx, train_seed, cv_seed, file_name_params=file_name_params)

            # Print Total Losses
            utils.print_total_loss(loss_train, loss_valid, loss_train_w, loss_valid_w)

            # Print and Get Accuracies of CV
            acc_train, acc_valid, acc_train_era, acc_valid_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train,
                                             prob_valid, y_valid, e_valid, show_accuracy)

            # Save Final Losses to File
            utils.save_final_loss_log(loss_log_path + self.model_name + '_', parameters, n_valid, n_cv,
                                      loss_train, loss_valid, loss_train_w, loss_valid_w,
                                      train_seed, cv_seed, acc_train, acc_train_era)

            # Save Loss Log to csv File
            if save_csv_log:
                self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                                  loss_train_w, loss_valid_w, acc_train, train_seed,
                                  cv_seed, n_valid, n_cv, parameters, file_name_params=file_name_params)

            # Return Final Result
            if return_prob_test:
                return prob_test


def grid_search(log_path, tr_x, tr_y, tr_e, clf, n_valid, n_cv, n_era, cv_seed, params, params_grid):
    """
         Grid Search
    """
    start_time = time.time()

    grid_search_model = GridSearchCV(estimator=clf,
                                     param_grid=params_grid,
                                     scoring='neg_log_loss',
                                     verbose=1,
                                     n_jobs=-1,
                                     cv=CrossValidation.era_k_fold_split(e=tr_e, n_valid=n_valid,
                                                                         n_cv=n_cv, n_era=n_era, seed=cv_seed))

    # Start Grid Search
    print('Grid Searching...')

    grid_search_model.fit(tr_x, tr_y, tr_e)

    best_parameters = grid_search_model.best_estimator_.get_params()
    best_score = grid_search_model.best_score_

    print('Best score: %0.6f' % best_score)
    print('Best parameters set:')

    for param_name in sorted(params_grid.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    total_time = time.time() - start_time

    utils.save_grid_search_log(log_path, params, params_grid, best_score, best_parameters, total_time)


if __name__ == '__main__':

    pass
