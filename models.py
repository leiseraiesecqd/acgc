import time
import utils
import os
from os.path import isdir
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from cross_validation import CrossValidation
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

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        # Training Model
        clf.fit(x_train, y_train, sample_weight=w_train)

        return clf

    def stack_fit(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid, w_valid, x_g_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        # Training Model
        clf.fit(x_train, y_train, sample_weight=w_train)

        return clf

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        # Get Classifier
        clf = self.get_clf(parameters)

        # Training Model
        if use_weight is True:
            clf.fit(x_train, y_train, sample_weight=w_train)
        else:
            clf.fit(x_train, y_train)

        return clf

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
        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    @staticmethod
    def print_loss(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid):

        loss_train, loss_valid, loss_train_w, loss_valid_w = \
            utils.print_loss_proba(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

        return loss_train, loss_valid, loss_train_w, loss_valid_w

    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, n_valid=4, n_cv=20, n_era=20,
              train_seed=None, cv_seed=None, era_list=None, parameters=None, show_importance=False, show_accuracy=False, 
              save_final_pred=True, save_cv_pred=True, save_final_prob_train=False, save_cv_prob_train=False,
              save_csv_log=True, csv_idx=None, cv_generator=None, return_prob_test=False, auto_train_pred_path=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        # Print Start Information and Get Model Name
        self.print_start_info()

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        # Get Cross Validation Generator
        if cv_generator is None:
            cv_generator = CrossValidation.era_k_fold_with_weight

        # Training on Cross Validation Sets
        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era \
                in cv_generator(x=self.x_train, y=self.y_train, w=self.w_train, e=self.e_train,
                                n_valid=n_valid, n_cv=n_cv, n_era=n_era, seed=cv_seed, era_list=era_list):

            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(count, n_cv))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            # Fitting and Training Model
            clf = self.fit(x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters)

            # Feature Importance
            if show_importance is True:
                self.get_importance(clf)

            # Prediction
            if save_cv_pred is True:
                cv_pred_path = pred_path + 'cv_results/' + self.model_name + '_cv_{}_'.format(count)
            else:
                cv_pred_path = None
            prob_test = self.predict(clf, self.x_test, pred_path=cv_pred_path)

            # Save Train Probabilities to CSV File
            if save_cv_prob_train is True:
                cv_prob_train_path = pred_path + 'cv_prob_train/' + self.model_name + '_cv_{}_'.format(count)
            else:
                cv_prob_train_path = None
            prob_train = self.get_prob_train(clf, self.x_train, pred_path=cv_prob_train_path)

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                self.print_loss(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid, show_accuracy)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + self.model_name + '_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, train_seed, cv_seed,
                                acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
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

        # Save Final Result
        if auto_train_pred_path is None:
            if save_final_pred is True:
                utils.save_pred_to_csv(pred_path + 'final_results/' + self.model_name + '_',
                                       self.id_test, prob_test_mean)
            if save_final_prob_train is True:
                utils.save_prob_train_to_csv(pred_path + 'final_prob_train/' + self.model_name + '_',
                                             prob_train_mean, self.y_train)
        elif save_final_pred is True:
            utils.save_pred_to_csv(auto_train_pred_path + self.model_name + '_' + str(csv_idx)
                                   + '_t' + str(train_seed) + '_c' + str(cv_seed) + '_',
                                   self.id_test, prob_test_mean)

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
        if save_csv_log is True:
            utils.save_final_loss_log_to_csv(csv_idx, csv_log_path + self.model_name + '_',
                                             loss_train_w_mean, loss_valid_w_mean, acc_train,
                                             train_seed, cv_seed, n_valid, n_cv, parameters)

        # Return Final Result
        if return_prob_test is True:
            return prob_test_mean

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        # Print Start Information and Get Model Name
        self.print_start_info()

        # Fitting and Training Model
        clf = self.stack_fit(x_train, y_train, w_train, x_g_train,
                             x_valid, y_valid, w_valid, x_g_valid, parameters)

        # Feature Importance
        if show_importance is True:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, loss_train_w, loss_valid_w =\
            self.print_loss(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

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
            if show_importance is True:
                self.get_importance(clf)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path +
                                     'cv_results/' + self.model_name + '_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train, pred_path=pred_path
                                             + 'cv_prob_train/' + self.model_name + '_cv_{}_'.format(count))

            # Print LogLoss
            print('------------------------------------------------------')
            loss_train, loss_valid, loss_train_w, loss_valid_w = self.print_loss(clf, x_train, y_train, w_train,
                                                                                 x_valid, y_valid, w_valid)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
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

    def stack_fit(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid, w_valid, x_g_valid, parameters=None):

        d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)

        # Booster
        eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]
        bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

        return bst

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        if use_weight is True:
            d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)
        else:
            d_train = xgb.DMatrix(x_train, label=y_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid)

        # Booster
        eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]
        bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

        return bst

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
        print('Predicting...')

        prob_test = model.predict(xgb.DMatrix(x_test))

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, model, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting...')

        prob_train = model.predict(xgb.DMatrix(x_train))

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    @staticmethod
    def print_loss(bst, x_train, y_train, w_train, x_valid, y_valid, w_valid):

        loss_train, loss_valid, loss_train_w, loss_valid_w = \
            utils.print_loss_xgb(bst, x_train, y_train, w_train, x_valid, y_valid, w_valid)

        return loss_train, loss_valid, loss_train_w, loss_valid_w


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

    def stack_fit(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid, w_valid, x_g_valid, parameters=None):

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

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Create Dataset
        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
        d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)

        # Booster
        bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                        valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

        return bst

    def stack_fit(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid, w_valid, x_g_valid, parameters=None):

        idx_category = [x_g_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        # Use Category
        d_train = lgb.Dataset(x_g_train, label=y_train, weight=w_train, categorical_feature=idx_category)
        d_valid = lgb.Dataset(x_g_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)

        # Booster
        bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                        valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

        return bst

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        if use_weight is True:
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
        print('Predicting...')

        prob_test = bst.predict(x_test)

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, bst, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting...')

        prob_train = bst.predict(x_train)

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    @staticmethod
    def print_loss(bst, x_train, y_train, w_train, x_valid, y_valid, w_valid):

        loss_train, loss_valid, loss_train_w, loss_valid_w = \
            utils.print_loss_lgb(bst, x_train, y_train, w_train, x_valid, y_valid, w_valid)

        return loss_train, loss_valid, loss_train_w, loss_valid_w

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        _ = self.print_start_info()

        # Training Model
        bst = self.stack_fit(x_train, y_train, w_train, x_g_train,
                             x_valid, y_valid, w_valid, x_g_valid, parameters)

        # Feature Importance
        if show_importance is True:
            self.get_importance(bst)

        # Print LogLoss
        print('------------------------------------------------------')
        loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_lgb(bst, x_train, y_train, w_train,
                                                                                  x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(bst, x_g_valid)
        prob_test = self.predict(bst, x_g_test)

        return prob_valid, prob_test, losses

    def prejudge_train_multiclass(self, pred_path=None, n_splits=10, n_cv=10, n_era=20, cv_seed=None, use_weight=True,
                                  parameters=None, show_importance=False, show_accuracy=False, cv_generator=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path)

        cv_counter = 0
        prob_test_total = np.array([])
        prob_train_total = np.array([])

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

            if use_weight is True:
                d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
                d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)
            else:
                d_train = lgb.Dataset(x_train, label=y_train, categorical_feature=idx_category)
                d_valid = lgb.Dataset(x_valid, label=y_valid, categorical_feature=idx_category)

            # Booster
            bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                            valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

            # Feature Importance
            if show_importance is True:
                self.get_importance(bst)

            # Prediction
            prob_test = self.predict(bst, self.x_test)

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(bst, self.x_train)

            # TODO: Print LogLoss

            if cv_counter == 1:
                prob_test_total = prob_test.reshape(-1, 1, n_era)
                prob_train_total = prob_train.reshape(-1, 1, n_era)
            else:
                np.concatenate((prob_test_total, prob_test.reshape(-1, 1, n_era)), axis=1)
                np.concatenate((prob_train_total, prob_train.reshape(-1, 1, n_era)), axis=1)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(prob_test_total, axis=1)
        prob_train_mean = np.mean(prob_train_total, axis=1)

        # TODO: Print Total Losses

        # TODO: Print and Get Accuracies of CV of All Train Set

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

    @staticmethod
    def stack_fit(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid, w_valid, x_g_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        idx_category = [x_g_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))

        # Fitting and Training Model
        clf.fit(x_g_train, y_train, sample_weight=w_train, categorical_feature=idx_category,
                eval_set=[(x_g_train, y_train), (x_g_valid, y_valid)], eval_names=['train', 'eval'],
                early_stopping_rounds=100, eval_sample_weight=[w_train, w_valid],
                eval_metric='logloss', verbose=True)

        return clf


class CatBoost(ModelBase):
    """
        CatBoost
    """
    @staticmethod
    def get_clf(parameters=None):

        print('Initialize Model...')
        clf = CatBoostClassifier(**parameters)

        return clf

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training CatBoost...')
        print('------------------------------------------------------')

        self.model_name = 'cb'

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        idx_category = [x_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        # Convert Zeros in Weights to Small Positive Numbers
        w_train = [0.001 if w == 0 else w for w in w_train]

        # Fitting and Training Model
        clf.fit(X=x_train, y=y_train, cat_features=idx_category, sample_weight=w_train,
                baseline=None, use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)

        return clf

    def stack_fit(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid, w_valid, x_g_valid, parameters=None):

        # Get Classifier
        clf = self.get_clf(parameters)

        idx_category = [x_g_train.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))
        print('------------------------------------------------------')

        # Convert Zeros in Weights to Small Positive Numbers
        w_train = [0.001 if w == 0 else w for w in w_train]

        # Fitting and Training Model
        clf.fit(X=x_g_train, y=y_train, cat_features=idx_category, sample_weight=w_train,
                baseline=None, use_best_model=None, eval_set=(x_g_valid, y_valid), verbose=True, plot=False)

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
        if use_weight is True:
            clf.fit(X=x_train, y=y_train, cat_features=idx_category, sample_weight=w_train,
                    baseline=None, use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)
        else:
            clf.fit(X=x_train, y=y_train, cat_features=idx_category, baseline=None,
                    use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)

        return clf


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
        self.dnn_seed = parameters['seed']
        self.display_step = parameters['display_step']
        self.save_path = parameters['save_path']
        self.log_path = parameters['log_path']

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Deep Neural Networks...')
        print('------------------------------------------------------')

        self.model_name = 'dnn'

    # Input Tensors
    def input_tensor(self):

        feature_num = self.x_train.shape[1]

        inputs_ = tf.placeholder(tf.float64, [None, feature_num], name='inputs')
        labels_ = tf.placeholder(tf.float64, None, name='labels')
        loss_weights_ = tf.placeholder(tf.float64, None, name='loss_weights')
        learning_rate_ = tf.placeholder(tf.float64, name='learning_rate')
        keep_prob_ = tf.placeholder(tf.float64, name='keep_prob')
        is_train_ = tf.placeholder(tf.bool, name='is_train')

        return inputs_, labels_, loss_weights_, learning_rate_, keep_prob_, is_train_

    # Full Connected Layer
    def fc_layer(self, x_tensor, layer_name, num_outputs, keep_prob, training):

        with tf.name_scope(layer_name):

            # x_shape = x_tensor.get_shape().as_list()
            #
            # weights = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs], dtype=tf.float64,
            #                                           stddev=2.0 / np.sqrt(x_shape[1])))
            #
            # biases = tf.Variable(tf.zeros([num_outputs], dtype=tf.float64))
            #
            # fc_layer = tf.add(tf.matmul(x_tensor, weights), biases)
            #
            # # Batch Normalization
            # fc_layer = tf.layers.batch_normalization(fc_layer, training=training)
            #
            # # Activate function
            # fc = tf.sigmoid(fc_layer)

            fc = tf.contrib.layers.fully_connected(x_tensor,
                                                   num_outputs,
                                                   activation_fn=tf.nn.sigmoid,
                                                   # weights_initializer=tf.truncated_normal_initializer(
                                                   # stddev=2.0 / math.sqrt(x_shape[1])),
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64,
                                                                                                            seed=self.dnn_seed),
                                                   # normalizer_fn=tf.contrib.layers.batch_norm,
                                                   biases_initializer=tf.zeros_initializer(dtype=tf.float64)
                                                   )

            tf.summary.histogram('fc_layer', fc)

            fc = tf.nn.dropout(fc, keep_prob)

        return fc

    # Output Layer
    def output_layer(self, x_tensor, layer_name, num_outputs):

        with tf.name_scope(layer_name):
            #  x_shape = x_tensor.get_shape().as_list()
            #
            #  weights = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs], stddev=2.0 / math.sqrt(x_shape[1])))
            #
            #  biases = tf.Variable(tf.zeros([num_outputs]))
            #
            #  with tf.name_scope('Wx_plus_b'):
            #      output_layer = tf.add(tf.matmul(x_tensor, weights), biases)
            #  tf.summary.histogram('output', output_layer)

            out = tf.contrib.layers.fully_connected(x_tensor,
                                                    num_outputs,
                                                    activation_fn=None,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64,
                                                                                                             seed=self.dnn_seed),
                                                    biases_initializer=tf.zeros_initializer())

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
            ones = tf.ones_like(y, dtype=tf.float64)
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
    def get_prob(self, sess, logits, x, batch_num, inputs, keep_prob, is_train):

        logits_pred = np.array([])

        for x_batch in self.get_batches_for_predict(x, batch_num):

            logits_pred_batch = sess.run(logits, {inputs: x_batch, keep_prob: 1.0, is_train: False})
            logits_pred_batch = logits_pred_batch.flatten()
            logits_pred = np.concatenate((logits_pred, logits_pred_batch))

        prob = 1.0 / (1.0 + np.exp(-logits_pred))

        return prob

    # Training
    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, n_valid=4, n_cv=20, n_era=20,
              train_seed=None, cv_seed=None, era_list=None, parameters=None, show_importance=False, show_accuracy=False, 
              save_final_pred=True, save_cv_pred=True, save_cv_prob_train=False, save_final_prob_train=False,
              save_csv_log=True, csv_idx=None, cv_generator=None, return_prob_test=False, auto_train_pred_path=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

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

            # Get Cross Validation Generator
            if cv_generator is None:
                cv_generator = CrossValidation.era_k_fold_with_weight

            for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era \
                    in cv_generator(self.x_train, self.y_train, self.w_train, self.e_train,
                                    n_valid=n_valid, n_cv=n_cv, n_era=n_era, seed=cv_seed, era_list=era_list):

                cv_counter += 1

                print('======================================================')
                print('Training on the Cross Validation Set: {}'.format(cv_counter))
                print('Validation Set Era: ', valid_era)
                print('------------------------------------------------------')

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
                                            is_train: True})

                        if str(cost) == 'nan':
                            raise ValueError('NaN BUG!!! Try Another Seed!!!')

                        if batch_counter % self.display_step == 0 and batch_i > 0:

                            summary_train, cost_train = sess.run([merged, cost_],
                                                                 {inputs: batch_x,
                                                                  labels: batch_y,
                                                                  weights: batch_w,
                                                                  keep_prob: 1.0,
                                                                  is_train: False})
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
                                                                          is_train: False})

                                valid_writer.add_summary(summary_valid_i, batch_counter)

                                cost_valid_all.append(cost_valid_i)

                            cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                            total_time = time.time() - start_time

                            print('CV: {} |'.format(cv_counter),
                                  'Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                                  'Batch: {} |'.format(batch_counter),
                                  'Time: {:>3.2f}s |'.format(total_time),
                                  'Train_Loss: {:>.8f} |'.format(cost_train),
                                  'Valid_Loss: {:>.8f}'.format(cost_valid))

                # Save Model
                # print('Saving model...')
                # saver = tf.train.Saver()
                # saver.save(sess, self.save_path + 'model.' + self.version + '.ckpt')

                # Prediction
                print('------------------------------------------------------')
                print('Predicting...')
                prob_train = self.get_prob(sess, logits, x_train, self.batch_size, inputs, keep_prob, is_train)
                prob_valid = self.get_prob(sess, logits, x_valid, self.batch_size, inputs, keep_prob, is_train)
                prob_test = self.get_prob(sess, logits, self.x_test, self.batch_size, inputs, keep_prob, is_train)

                loss_train, loss_valid, \
                    loss_train_w, loss_valid_w = utils.print_loss_dnn(prob_train, prob_valid,
                                                                      y_train, w_train, y_valid, w_valid)
                prob_test_total.append(prob_test)
                prob_train_total.append(prob_train)
                loss_train_total.append(loss_train)
                loss_valid_total.append(loss_valid)
                loss_train_w_total.append(loss_train_w)
                loss_valid_w_total.append(loss_valid_w)

                # Print and Get Accuracies of CV
                acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                    utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid, show_accuracy)

                utils.save_loss_log(loss_log_path + self.model_name + '_', cv_counter, self.parameters, n_valid, n_cv,
                                    valid_era, loss_train, loss_valid, loss_train_w, loss_valid_w, train_seed, cv_seed,
                                    acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era)

                if save_cv_pred is True:
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

            # Save Final Result
            if auto_train_pred_path is None:
                if save_final_pred is True:
                    utils.save_pred_to_csv(pred_path + 'final_results/' + self.model_name + '_',
                                           self.id_test, prob_test_mean)
                if save_final_prob_train is True:
                    utils.save_prob_train_to_csv(pred_path + 'final_prob_train/' + self.model_name + '_',
                                                 prob_train_mean, self.y_train)
            elif save_final_pred is True:
                utils.save_pred_to_csv(auto_train_pred_path + self.model_name + '_' +
                                       str(csv_idx) + '_' + str(train_seed) + '_' + str(cv_seed) + '_',
                                       self.id_test, prob_test_mean)

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
            if save_csv_log is True:
                utils.save_final_loss_log_to_csv(csv_idx, csv_log_path + self.model_name + '_',
                                                 loss_train_w_mean, loss_valid_w_mean, acc_train,
                                                 train_seed, cv_seed, n_valid, n_cv, parameters)

            # Return Final Result
            if return_prob_test is True:
                return prob_test_mean

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters=None, show_importance=False):

        print('------------------------------------------------------')
        print('Training Deep Neural Network...')
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
            print('Predicting...')

            logits_pred_train = sess.run(logits, {inputs: x_train, keep_prob: 1.0, is_train: False})
            logits_pred_valid = sess.run(logits, {inputs: x_valid, keep_prob: 1.0, is_train: False})
            logits_pred_test = sess.run(logits, {inputs: x_test, keep_prob: 1.0, is_train: False})

            logits_pred_train = logits_pred_train.flatten()
            logits_pred_valid = logits_pred_valid.flatten()
            logits_pred_test = logits_pred_test.flatten()

            prob_train = 1.0 / (1.0 + np.exp(-logits_pred_train))
            prob_valid = 1.0 / (1.0 + np.exp(-logits_pred_valid))
            prob_test = 1.0 / (1.0 + np.exp(-logits_pred_test))

            loss_train, loss_valid, \
                loss_train_w, loss_valid_w = utils.print_loss_dnn(prob_train, prob_valid,
                                                                  y_train, w_train, y_valid, w_valid)

            losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

            return prob_valid, prob_test, losses


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

    utils.seve_grid_search_log(log_path, params, params_grid, best_score, best_parameters, total_time)


if __name__ == '__main__':

    pass
