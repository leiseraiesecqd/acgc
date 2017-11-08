import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from models import utils

from models.cross_validation import CrossValidation
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
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, use_multi_group=False):

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
        self.use_multi_group = use_multi_group

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

    def fit_with_round_log(self, boost_round_log_path, cv_count, x_train, y_train,
                           w_train, x_valid, y_valid, w_valid, parameters,
                           param_name_list, param_value_list, append_info=''):

        param_info = ''
        param_name = ''
        for i in range(len(param_name_list)):
            param_name += '_' + utils.get_simple_param_name(param_name_list[i])
            param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])

        boost_round_log_path += self.model_name + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_' + append_info + '/'
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

    def save_boost_round_log(self, boost_round_log_path, idx_round, train_loss_round_mean,
                             valid_loss_round_mean, train_seed, cv_seed, csv_idx, parameters,
                             param_name_list, param_value_list, append_info=''):

        param_info = ''
        param_name = ''
        for i in range(len(param_name_list)):
            param_name += '_' + utils.get_simple_param_name(param_name_list[i])
            param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])

        boost_round_log_path += self.model_name + '/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_' + append_info + '/'
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

    def save_csv_log(self, mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                     loss_train_w_mean, loss_valid_w_mean, acc_train, train_seed, cv_seed,
                     n_valid, n_cv, parameters, file_name_params=None, append_info=''):

        if mode == 'auto_grid_search':

            param_info = ''
            param_name = ''
            for i in range(len(param_name_list)):
                param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])
                param_name += '_' + utils.get_simple_param_name(param_name_list[i])

            csv_log_path += self.model_name + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + '_' + append_info + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + param_name + '/'
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
            csv_log_path += self.model_name + '_' + append_info + '/'
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

    def save_final_pred(self, mode, save_final_pred, prob_test_mean, pred_path, parameters,
                        csv_idx, train_seed, cv_seed, boost_round_log_path=None, param_name_list=None,
                        param_value_list=None, file_name_params=None, append_info=''):

        params = '_'
        if file_name_params is not None:
            for p_name in file_name_params:
                params += utils.get_simple_param_name(p_name) + '-' + str(parameters[p_name]) + '_'
        else:
            for p_name, p_value in parameters.items():
                params += utils.get_simple_param_name(p_name) + '-' + str(p_value) + '_'

        if save_final_pred:

            if mode == 'auto_train':

                pred_path += self.model_name + '/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + append_info + '/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + params + 'results/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + str(csv_idx) + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + '_'
                utils.save_pred_to_csv(pred_path, self.id_test, prob_test_mean)

            elif mode == 'auto_train_boost_round':

                param_info = ''
                param_name = ''
                for i in range(len(param_name_list)):
                    param_info += '_' + utils.get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])
                    param_name += '_' + utils.get_simple_param_name(param_name_list[i])

                boost_round_log_path += self.model_name + '/'
                utils.check_dir([boost_round_log_path])
                boost_round_log_path += self.model_name + '_' + append_info + '/'
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
                pred_path += 'final_results/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + append_info + '/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + params
                utils.check_dir([pred_path])
                utils.save_pred_to_csv(pred_path, self.id_test, prob_test_mean)

    @staticmethod
    def get_rescale_rate(y):

        positive = 0
        for y_ in y:
            if y_ == 1:
                positive += 1

        positive_rate = positive / len(y)
        rescale_rate = len(y) / (2*positive)

        return positive_rate, rescale_rate

    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, boost_round_log_path=None,
              train_seed=None, cv_args=None, parameters=None, show_importance=False, show_accuracy=False,
              save_cv_pred=True, save_cv_prob_train=False, save_final_pred=True, save_final_prob_train=False,
              save_csv_log=True, csv_idx=None, rescale=False, return_prob_test=False, mode=None,
              param_name_list=None, param_value_list=None, file_name_params=None, append_info=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        n_valid = cv_args['n_valid']
        n_cv = cv_args['n_cv']
        n_era = cv_args['n_era']
        cv_seed = cv_args['cv_seed']

        # Append Information
        if append_info is None:
            append_info = 'v-' + str(n_valid) + '_c-' + str(n_cv) + '_e-' + str(n_era)
            if 'window_size' in cv_args:
                append_info += '_w-' + str(cv_args['window_size'])

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
        if 'cv_generator' in cv_args:
            cv_generator = cv_args['cv_generator']
            if cv_generator is None:
                cv_generator = CrossValidation.era_k_fold
            cv_args.pop('cv_generator')
        else:
            cv_generator = CrossValidation.era_k_fold
        print('------------------------------------------------------')
        print('Using CV Generator: {}'.format(getattr(cv_generator, '__name__')))

        if 'era_list' in cv_args:
            print('Era List: ', cv_args['era_list'])
        if 'window_size' in cv_args:
            print('Window Size: ', cv_args['window_size'])
        if 'cv_weights' in cv_args:
            cv_weights = cv_args['cv_weights']
            cv_args.pop('cv_weights')
        else:
            cv_weights = None

        # Training on Cross Validation Sets
        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era \
                in cv_generator(x=self.x_train, y=self.y_train, w=self.w_train, e=self.e_train, **cv_args):

            # Get Positive Rate of Train Set and Rescale Rate
            positive_rate_train, rescale_rate = self.get_rescale_rate(y_train)
            positive_rate_valid, _ = self.get_rescale_rate(y_valid)

            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            print('Number of Features: ', x_train.shape[1])
            print('------------------------------------------------------')
            print('Positive Rate of Train Set: ', positive_rate_train)
            print('Positive Rate of Valid Set: ', positive_rate_valid)
            print('Rescale Rate of Valid Set: ', rescale_rate)
            print('------------------------------------------------------')

            # Fitting and Training Model
            if mode == 'auto_train_boost_round':
                clf, idx_round_cv, train_loss_round_cv, valid_loss_round_cv = \
                    self.fit_with_round_log(boost_round_log_path, cv_count, x_train, y_train, w_train, x_valid,
                                            y_valid, w_valid, parameters, param_name_list, param_value_list,
                                            append_info=append_info)

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
            if rescale:
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

        # Calculate Means of prob and losses
        prob_test_mean, prob_train_mean, loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean = \
            utils.calculate_means(prob_test_total, prob_train_total, loss_train_total, loss_valid_total,
                                  loss_train_w_total, loss_valid_w_total, weights=cv_weights)

        # Save Logs of num_boost_round
        if mode == 'auto_train_boost_round':
            train_loss_round_mean, valid_loss_round_mean = \
                utils.calculate_boost_round_means(train_loss_round_total, valid_loss_round_total, weights=cv_weights)
            self.save_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean, valid_loss_round_mean,
                                      train_seed, cv_seed, csv_idx, parameters, param_name_list, param_value_list,
                                      append_info=append_info)

        # Save 'num_boost_round'
        if self.model_name in ['xgb', 'lgb']:
            parameters['num_boost_round'] = self.num_boost_round

        # Save Final Result
        if save_final_pred:
            self.save_final_pred(mode, save_final_pred, prob_test_mean, pred_path, parameters, csv_idx,
                                 train_seed, cv_seed, boost_round_log_path, param_name_list, param_value_list,
                                 file_name_params=file_name_params, append_info=append_info)

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
                              loss_train_w_mean, loss_valid_w_mean, acc_train, train_seed, cv_seed,
                              n_valid, n_cv, parameters, file_name_params=file_name_params, append_info=append_info)

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
            cv_generator = CrossValidation.sk_k_fold

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
                             pred_path=None, loss_log_path=None, csv_log_path=None, parameters=None, cv_args=None,
                             train_seed=None, show_importance=False, show_accuracy=False, save_final_pred=True,
                             save_final_prob_train=False, save_csv_log=True, csv_idx=None, mode=None,
                             file_name_params=None, param_name_list=None, param_value_list=None, append_info=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        n_valid = cv_args['n_valid']
        n_cv = cv_args['n_cv']
        cv_seed = cv_args['cv_seed']

        # Append Information
        if append_info is None:
            append_info = 'v-' + str(n_valid) + '_c-' + str(n_cv)

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
                                 parameters, csv_idx, train_seed, cv_seed, file_name_params=file_name_params,
                                 append_info=append_info)

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
                              loss_train_w, loss_valid_w, acc_train, train_seed, cv_seed, n_valid, n_cv,
                              parameters, file_name_params=file_name_params, append_info=append_info)

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
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, num_boost_round, use_multi_group=False):

        super(XGBoost, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, id_te, use_multi_group)

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
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, num_boost_round, use_multi_group=False):

        super(LightGBM, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, id_te, use_multi_group)

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
        if self.use_multi_group:
            print('------------------------------------------------------')
            print('[W] Using Multi Groups...')
            idx_category = [x_train.shape[1] - 2, x_train.shape[1] - 1]
        else:
            print('------------------------------------------------------')
            print('[W] Using Single Group...')
            idx_category = [x_train.shape[1] - 1]
        print('------------------------------------------------------')
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
        d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)

        # Booster
        bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                        valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

        return bst

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        if self.use_multi_group:
            print('------------------------------------------------------')
            print('[W] Using Multi Groups...')
            idx_category = [x_train.shape[1] - 2, x_train.shape[1] - 1]
        else:
            print('------------------------------------------------------')
            print('[W] Using Single Group...')
            idx_category = [x_train.shape[1] - 1]
        print('------------------------------------------------------')
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
            cv_generator = CrossValidation.sk_k_fold
        print('------------------------------------------------------')
        print('Using CV Generator: {}'.format(getattr(cv_generator, '__name__')))

        # Cross Validation
        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in cv_generator(x=self.x_train, y=self.y_train, w=self.w_train,
                                                      n_splits=n_splits, n_cv=n_cv, seed=cv_seed):

            cv_counter += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(cv_counter, n_cv))

            idx_category = [x_train.shape[1] - 2, x_train.shape[1] - 1]
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

        if self.use_multi_group:
            print('------------------------------------------------------')
            print('[W] Using Multi Groups...')
            idx_category = [x_train.shape[1] - 2, x_train.shape[1] - 1]
        else:
            print('------------------------------------------------------')
            print('[W] Using Single Group...')
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

        if self.use_multi_group:
            print('------------------------------------------------------')
            print('[W] Using Multi Groups...')
            idx_category = [x_train.shape[1] - 2, x_train.shape[1] - 1]
        else:
            print('------------------------------------------------------')
            print('[W] Using Single Group...')
            idx_category = [x_train.shape[1] - 1]
        print('------------------------------------------------------')
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        # Convert Zeros in Weights to Small Positive Numbers
        w_train = [0.001 if w == 0 else w for w in w_train]

        # Fitting and Training Model
        clf.fit(X=x_train, y=y_train, cat_features=idx_category, sample_weight=w_train,
                baseline=None, use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)

        return clf

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        # Get Classifier
        clf = self.get_clf(parameters)

        if self.use_multi_group:
            print('------------------------------------------------------')
            print('[W] Using Multi Groups...')
            idx_category = [x_train.shape[1] - 2, x_train.shape[1] - 1]
        else:
            print('------------------------------------------------------')
            print('[W] Using Single Group...')
            idx_category = [x_train.shape[1] - 1]
        print('------------------------------------------------------')
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
