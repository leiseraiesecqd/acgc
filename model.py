#  import math
import time
import utils
import os
from os.path import isdir
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import cross_val_score
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
# from xgboost import plot_importance
# from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)
color = sns.color_palette()


# Logistic Regression

class LRegression:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])

    def get_importance(self, clf):

        self.importance = np.abs(clf.coef_)[0]
        indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, indices[f], self.importance[indices[f]]))

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Logistic Regression')
        plt.bar(range(feature_num), self.importance[self.indices], color=color[0], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def train(self, parameters):

        clf_lr = LogisticRegression(random_state=1)
        '''
        LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                           penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
                           verbose=0, warm_start=False)
        '''

        clf_lr.fit(self.x_train, self.y_train, sample_weight=self.w_train)

        # K-fold cross-validation on Logistic Regression
        scores = cross_val_score(clf_lr, self.x_train, self.y_train, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))

        self.get_importance(clf_lr)


# k-Nearest Neighbor

class KNearestNeighbor:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

    def train(self, parameters):

        clf_knn = KNeighborsClassifier()
        '''
        KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                             weights='uniform')
        '''

        clf_knn.fit(self.x_train, self.y_train)  # without parameter sample_weight

        # K-fold cross-validation on Logistic Regression
        scores = cross_val_score(clf_knn, self.x_train, self.y_train, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# SVM-SVC

class SupportVectorClustering:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

    def train(self, parameters):

        clf_svc = SVC(random_state=1)

        clf_svc.fit(self.x_train, self.y_train, sample_weight=self.w_train)

        scores = cross_val_score(clf_svc, self.x_train, self.y_train, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# Gaussian NB

class Gaussian:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

    def train(self, parameters):

        clf_gnb = GaussianNB()

        clf_gnb.fit(self.x_train, self.y_train, sample_weight=self.w_train)

        scores = cross_val_score(clf_gnb, self.x_train, self.y_train, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# Decision Tree

class DecisionTree:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self, parameters):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Decision Tree')
        plt.bar(range(feature_num), self.importance[self.indices], color=color[1], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def clf(self, parameters):

        clf = DecisionTreeClassifier(**parameters)

        return clf

    def predict(self, clf, pred_path):

        print('Predicting...')

        prob_test = clf.predict(self.x_test)

        utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def train(self, pred_path, parameters=None):

        count = 0
        prob_total = []

        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(x=self.x_train,
                                                                                     y=self.y_train,
                                                                                     w=self.w_train,
                                                                                     e=self.e_train):
            count += 1

            print('===========================================')
            print('Training on the Cross Validation Set: {}'.format(count))

            clf_dt = self.clf(parameters)
            '''
            DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                   max_features=None, max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, presort=False, random_state=1,
                                   splitter='best')
            '''

            clf_dt.fit(x_train, y_train, sample_weight=w_train)

            scores = clf_dt.score(x_valid, y_valid, sample_weight=w_valid)

            print('mean accuracy on validation set: %0.6f' % scores)

            self.get_importance(clf_dt)

            prob_test = self.predict(clf_dt, pred_path + 'dt_cv_{}_'.format(count))
            prob_total.append(list(prob_test))

        print('===========================================')
        print('Calculating final result...')

        prob_mean = np.mean(np.array(prob_total), axis=0)

        utils.save_pred_to_csv(pred_path + 'dt_', self.id_test, prob_mean)


# Random Forest

class RandomForest:

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

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]
        self.std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Random Forest')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[2], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def clf(self, parameters):

        clf = RandomForestClassifier(**parameters)

        return clf

    def predict(self, clf, pred_path):

        print('Predicting...')

        prob_test = clf.predict(self.x_test)

        utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def train(self, pred_path, parameters=None):

        count = 0
        prob_total = []

        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(x=self.x_train,
                                                                                     y=self.y_train,
                                                                                     w=self.w_train,
                                                                                     e=self.e_train):
            count += 1

            print('===========================================')
            print('Training on the Cross Validation Set: {}'.format(count))

            clf_rf = self.clf(parameters)
            '''
            RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                   max_depth=None, max_features='auto', max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                   oob_score=False, random_state=1, verbose=1, warm_start=False)
            '''

            clf_rf.fit(x_train, y_train, sample_weight=w_train)

            scores = clf_rf.score(x_valid, y_valid, sample_weight=w_valid)

            print('mean accuracy on validation set: %0.6f' % scores)

            self.get_importance(clf_rf)

            prob_test = self.predict(clf_rf, pred_path + 'rf_cv_{}_'.format(count))
            prob_total.append(list(prob_test))

        print('===========================================')
        print('Calculating final result...')

        prob_mean = np.mean(np.array(prob_total), axis=0)

        utils.save_pred_to_csv(pred_path + 'rf_', self.id_test, prob_mean)


# Extra Trees

class ExtraTrees:

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

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]
        self.std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Extra Trees')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[3], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def clf(self, parameters):

        clf = ExtraTreesClassifier(**parameters)

        return clf

    def predict(self, clf, pred_path):

        print('Predicting...')

        prob_test = clf.predict(self.x_test)

        utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def train(self, pred_path, parameters=None):

        count = 0
        prob_total = []

        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(x=self.x_train,
                                                                                     y=self.y_train,
                                                                                     w=self.w_train,
                                                                                     e=self.e_train):
            count += 1

            print('===========================================')
            print('Training on the Cross Validation Set: {}'.format(count))

            clf_et = self.clf(parameters)
            '''
            ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                                 max_depth=None, max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                 oob_score=False, random_state=1, verbose=1, warm_start=False)
            '''

            clf_et.fit(x_train, y_train, sample_weight=w_train)

            scores = clf_et.score(x_valid, y_valid, sample_weight=w_valid)

            print('mean accuracy on validation set: %0.6f' % scores)

            self.get_importance(clf_et)

            prob_test = self.predict(clf_et, pred_path + 'et_cv_{}_'.format(count))
            prob_total.append(list(prob_test))

        print('===========================================')
        print('Calculating final result...')

        prob_mean = np.mean(np.array(prob_total), axis=0)

        utils.save_pred_to_csv(pred_path + 'et_', self.id_test, prob_mean)


# AdaBoost

class AdaBoost:

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

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]
        self.std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in AdaBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[4], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def clf(self, parameters):

        clf = AdaBoostClassifier(**parameters)

        return clf

    def predict(self, clf, pred_path):

        print('Predicting...')

        prob_test = clf.predict(self.x_test)

        utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def train(self, pred_path, parameters=None):

        count = 0
        prob_total = []

        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(x=self.x_train,
                                                                                     y=self.y_train,
                                                                                     w=self.w_train,
                                                                                     e=self.e_train):
            count += 1

            print('===========================================')
            print('Training on the Cross Validation Set: {}'.format(count))

            clf_ab = self.clf(parameters)
            '''
            AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                               learning_rate=1.0, n_estimators=50, random_state=1)
            '''

            clf_ab.fit(x_train, y_train, sample_weight=w_train)

            scores = clf_ab.score(x_valid, y_valid, sample_weight=w_valid)

            print('mean accuracy on validation set: %0.6f' % scores)

            self.get_importance(clf_ab)

            prob_test = self.predict(clf_ab, pred_path + 'ab_cv_{}_'.format(count))
            prob_total.append(list(prob_test))

        print('===========================================')
        print('Calculating final result...')

        prob_mean = np.mean(np.array(prob_total), axis=0)

        utils.save_pred_to_csv(pred_path + 'ab_', self.id_test, prob_mean)


# GradientBoosting

class GradientBoosting:

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

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]
        self.std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print('Importance:')
            print('%d. feature %d (%f)' % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in GradientBoosting')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[5], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def clf(self, parameters):

        clf = GradientBoostingClassifier(**parameters)

        return clf

    def predict(self, clf, pred_path):

        print('Predicting...')

        prob_test = clf.predict(self.x_test)

        utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def train(self, pred_path, parameters=None):

        count = 0
        prob_total = []

        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(x=self.x_train,
                                                                                     y=self.y_train,
                                                                                     w=self.w_train,
                                                                                     e=self.e_train):
            count += 1

            print('===========================================')
            print('Training on the Cross Validation Set: {}'.format(count))

            clf_gb = self.clf(parameters)
            '''
            GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                       learning_rate=0.1, loss='deviance', max_depth=3,
                                       max_features=None, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=2,
                                       min_weight_fraction_leaf=0.0, n_estimators=100,
                                       presort='auto', random_state=1, subsample=1.0, verbose=1,
                                       warm_start=False)
            '''

            clf_gb.fit(x_train, y_train, sample_weight=w_train)

            scores = clf_gb.score(x_valid, y_valid, sample_weight=w_valid)

            print('mean accuracy on validation set: %0.6f' % scores)

            self.get_importance(clf_gb)

            prob_test = self.predict(clf_gb, pred_path + 'gb_cv_{}_'.format(count))
            prob_total.append(list(prob_test))

        print('===========================================')
        print('Calculating final result...')

        prob_mean = np.mean(np.array(prob_total), axis=0)

        utils.save_pred_to_csv(pred_path + 'gb_', self.id_test, prob_mean)


# XGBoost

class XGBoost:

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

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in XGBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[6], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def clf(self, parameters=None):

        '''
            class xgboost.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                                        objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None,
                                        gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                                        colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                                        scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,
                                        missing=None, **kwargs)
        '''
        # clf = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
        #                     gamma=2, learning_rate=0.05, max_delta_step=0, max_depth=3,
        #                     min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
        #                     objective='binary:logistic', reg_alpha=0, reg_lambda=1,
        #                     scale_pos_weight=1, seed=0, silent=True, subsample=0.8)

        print('Initialize Model...')

        clf = XGBClassifier(**parameters)

        return clf

    def predict(self, model, pred_path):

        print('Predicting...')

        prob_test = model.predict(xgb.DMatrix(self.x_test))

        utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def train(self, pred_path, parameters=None):

        # sk-learn module

        # clf_xgb = self.clf()
        #
        # train_scores = cross_val_score(clf_xgb, self.x_train, self.y_train, cv=20)
        # print("Accuracy: %0.6f (+/- %0.6f)" % (train_scores.mean(), train_scores.std() * 2))
        #
        # count = 0
        #
        # for x_train, y_train, w_train, \
        #     x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(self.x_train,
        #                                                                              self.y_train,
        #                                                                              self.w_train):
        #
        #     count += 1
        #     print('Training CV: {}'.format(count))
        #
        #     clf_xgb.fit(x_train, y_train, sample_weight=w_train,
        #                 eval_set=[(x_train, y_train), (x_valid, y_valid)],
        #                 eval_metric='logloss', verbose=True)
        #
        #     result = clf_xgb.evals_result()
        #
        #     print(result)
        #
        #     self.prediction(clf_xgb)
        #
        #     self.get_importance(clf_xgb)

        count = 0
        prob_total = []

        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(x=self.x_train,
                                                                                     y=self.y_train,
                                                                                     w=self.w_train,
                                                                                     e=self.e_train):

            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))

            d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)

            # parameters = {'learning_rate': 0.05, 'n_estimators': 1000, 'max_depth': 10,
            #               'min_child_weight': 5, 'gamma': 0, 'silent': 1, 'objective': 'binary:logistic',
            #               'early_stopping_rounds': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
            #               'eval_metric': 'logloss'}

            eval_list = [(d_valid, 'eval'), (d_train, 'train')]

            bst = xgb.train(parameters, d_train, num_boost_round=30, evals=eval_list)

            # Prediction
            prob_test = self.predict(bst, pred_path + 'xgb_cv_{}_'.format(count))
            prob_total.append(list(prob_test))

        print('======================================================')
        print('Calculating final result...')

        prob_mean = np.mean(np.array(prob_total), axis=0)

        utils.save_pred_to_csv(pred_path + 'xgb_', self.id_test, prob_mean)


# LightGBM

class LightGBM:

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

    def get_importance(self, clf):

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in XGBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[6], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def clf(self, parameters=None):

        '''
            class xgboost.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                                        objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None,
                                        gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                                        colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                                        scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,
                                        missing=None, **kwargs)
        '''
        # clf = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
        #                     gamma=2, learning_rate=0.05, max_delta_step=0, max_depth=3,
        #                     min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
        #                     objective='binary:logistic', reg_alpha=0, reg_lambda=1,
        #                     scale_pos_weight=1, seed=0, silent=True, subsample=0.8)

        print('Initialize Model...')

        clf = LGBMClassifier(**parameters)

        return clf

    def predict(self, model, pred_path):

        print('Predicting...')

        prob_test = model.predict(self.x_test)

        utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def train(self, pred_path, parameters=None):

        # sk-learn module

        # clf_lgb = self.clf()
        #
        # train_scores = cross_val_score(clf_lgb, self.x_train, self.y_train, cv=20)
        # print("Accuracy: %0.6f (+/- %0.6f)" % (train_scores.mean(), train_scores.std() * 2))
        #
        # count = 0
        #
        # for x_train, y_train, w_train, \
        #     x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(self.x_train,
        #                                                                              self.y_train,
        #                                                                              self.w_train
        #                                                                              self.e_train):
        #
        #     count += 1
        #     print('Training CV: {}'.format(count))
        #
        #     clf_lgb.fit(x_train, y_train, sample_weight=w_train,
        #                 eval_set=[(x_train, y_train), (x_valid, y_valid)],
        #                 eval_metric='logloss', verbose=True)
        #
        #     result = clf_lgb.evals_result()
        #
        #     print(result)
        #
        #     self.prediction(clf_lgb)
        #
        #     self.get_importance(clf_lgb)

        count = 0
        prob_total = []

        for x_train, y_train, w_train, \
            x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(x=self.x_train,
                                                                                     y=self.y_train,
                                                                                     w=self.w_train,
                                                                                     e=self.e_train):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))

            d_train = lgb.Dataset(x_train, label=y_train, weight=w_train)
            d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid)

            # parameters = {'learning_rate': 0.05, 'n_estimators': 1000, 'max_depth': 10,
            #               'min_child_weight': 5, 'gamma': 0, 'silent': 1, 'objective': 'binary:logistic',
            #               'early_stopping_rounds': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
            #               'eval_metric': 'logloss'}

            bst = lgb.train(parameters, d_train, num_boost_round=30,
                            valid_sets=[d_valid, d_train], valid_names=['eval', 'train'])

            # Prediction
            prob_test = self.predict(bst, pred_path + 'lgb_cv_{}_'.format(count))
            prob_total.append(list(prob_test))

        print('======================================================')
        print('Calculating final result...')

        prob_mean = np.mean(np.array(prob_total), axis=0)

        utils.save_pred_to_csv(pred_path + 'lgb_', self.id_test, prob_mean)


# Deep Neural Networks

class DeepNeuralNetworks:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, parameters):

        # Inputs
        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

        # Hyperparameters
        self.version = parameters['version']
        self.epochs = parameters['epochs']
        self.layers_number = parameters['layers_number']
        self.unit_number = parameters['unit_number']
        self.learning_rate = parameters['learning_rate']
        self.keep_probability = parameters['keep_probability']
        self.batch_size = parameters['batch_size']
        self.display_step = parameters['display_step']
        self.save_path = parameters['save_path']
        self.log_path = parameters['log_path']

    # Input Tensors
    def input_tensor(self, n_feature):

        inputs_ = tf.placeholder(tf.float64, [None, n_feature], name='inputs')
        labels_ = tf.placeholder(tf.float64, None, name='labels')
        loss_weights_ = tf.placeholder(tf.float64, None, name='loss_weights')
        learning_rate_ = tf.placeholder(tf.float64, name='learning_rate')
        keep_prob_ = tf.placeholder(tf.float64, name='keep_prob')
        is_train_ = tf.placeholder(tf.bool, name='is_train')

        return inputs_, labels_, loss_weights_, learning_rate_, keep_prob_, is_train_

    # Full Connected Layer
    def fc_layer(self, x_tensor, layer_name, num_outputs, keep_prob, training):

        with tf.name_scope(layer_name):

            #  x_shape = x_tensor.get_shape().as_list()

            # weights = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs], stddev=2.0 / math.sqrt(x_shape[1])))
            #
            # biases = tf.Variable(tf.zeros([num_outputs]))

            # fc_layer = tf.add(tf.matmul(x_tensor, weights), biases)
            #
            # Batch Normalization
            # fc_layer = tf.layers.batch_normalization(fc_layer, training=training)
            #
            # Activate function
            # fc = tf.nn.relu(fc_layer)
            # fc = tf.nn.elu(fc_layer)

            fc = tf.contrib.layers.fully_connected(x_tensor,
                                                   num_outputs,
                                                   activation_fn=tf.nn.sigmoid,
                                                   # weights_initializer=tf.truncated_normal_initializer(
                                                   # stddev=2.0 / math.sqrt(x_shape[1])),
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64),
                                                   biases_initializer=tf.zeros_initializer())

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
                                                    activation_fn=None)

        return out

    # Model
    def model(self, x, n_layers, n_unit, keep_prob, is_training):

        #  fc1 = fc_layer(x, 'fc1', n_unit[0], keep_prob)
        #  fc2 = fc_layer(fc1, 'fc2', n_unit[1], keep_prob)
        #  fc3 = fc_layer(fc2, 'fc3', n_unit[2], keep_prob)
        #  fc4 = fc_layer(fc3, 'fc4', n_unit[3], keep_prob)
        #  fc5 = fc_layer(fc4, 'fc5', n_unit[4], keep_prob)

        #  logit_ = self.output_layer(fc5, 'output', 1)

        fc = []
        fc.append(x)

        for i in range(n_layers):
            fc.append(self.fc_layer(fc[i], 'fc{}'.format(i + 1), n_unit[i], keep_prob, is_training))

        logit_ = self.output_layer(fc[n_layers], 'output', 1)

        return logit_

    # LogLoss
    def log_loss(self, logit, weight, label):

        with tf.name_scope('prob'):
            prob = tf.nn.sigmoid(logit)

            #  with tf.name_scope('weight'):
            weight = weight / tf.reduce_sum(weight)

        with tf.name_scope('logloss'):
            #  loss = tf.losses.log_loss(labels=label, predictions=prob, weights=weight)
            loss = - tf.reduce_sum(weight * (label * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) + (1 - label) * tf.log(
                tf.clip_by_value((1 - prob), 1e-10, 1.0))))

        tf.summary.scalar('logloss', loss)

        return loss

    # Get Batches
    def get_batches(self, x, y, w, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num, batch_num):

            if ii != n_batches * batch_num:
                batch_x, batch_y, batch_w = x[ii: ii + batch_num], y[ii: ii + batch_num], w[ii: ii + batch_num]

            else:
                batch_x, batch_y, batch_w = x[ii:], y[ii:], w[ii:]

            yield batch_x, batch_y, batch_w

    # Training
    def train(self, pred_path):

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            feature_num = list(self.x_train.shape)[1]
            inputs, labels, loss_weights, lr, keep_prob, is_train = self.input_tensor(feature_num)

            # Logits
            logits = self.model(inputs, self.layers_number, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                #  cost_ = self.log_loss(logits, loss_weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

            # LogLoss
            #  with tf.name_scope('LogLoss'):
            #      logloss = log_loss(logits, loss_weights, labels)

        # Training
        print('Training...')

        with tf.Session(graph=train_graph) as sess:

            # Merge all the summaries
            merged = tf.summary.merge_all()

            start_time = time.time()

            cv_counter = 0

            prob_total = []

            for x_train, y_train, w_train, \
                x_valid, y_valid, w_valid in CrossValidation.sk_group_k_fold_with_weight(self.x_train,
                                                                                         self.y_train,
                                                                                         self.w_train,
                                                                                         self.e_train):

                cv_counter += 1

                print('======================================================================================================')
                print('Training on the Cross Validation Set: {}'.format(cv_counter))

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
                                            loss_weights: batch_w,
                                            lr: self.learning_rate,
                                            keep_prob: self.keep_probability,
                                            is_train: True})

                        if batch_counter % self.display_step == 0 and batch_i > 0:

                            summary_train, cost_train = sess.run([merged, cost_],
                                                                 {inputs: batch_x,
                                                                  labels: batch_y,
                                                                  loss_weights: batch_w,
                                                                  keep_prob: 1.0,
                                                                  is_train: False})
                            train_writer.add_summary(summary_train, batch_counter)

                            cost_valid_a = []

                            for iii, (valid_batch_x,
                                      valid_batch_y,
                                      valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                                   y_valid,
                                                                                   w_valid,
                                                                                   self.batch_size)):

                                summary_valid_i, cost_valid_i = sess.run([merged, cost_],
                                                                         {inputs: valid_batch_x,
                                                                          labels: valid_batch_y,
                                                                          loss_weights: valid_batch_w,
                                                                          keep_prob: 1.0,
                                                                          is_train: False})

                                valid_writer.add_summary(summary_valid_i, batch_counter)

                                cost_valid_a.append(cost_valid_i)

                            cost_valid = sum(cost_valid_a) / len(cost_valid_a)

                            total_time = time.time() - start_time

                            print("CV: {} |".format(cv_counter),
                                  "Epoch: {}/{} |".format(epoch_i + 1, self.epochs),
                                  "Batch: {} |".format(batch_counter),
                                  "Time: {:>3.2f}s |".format(total_time),
                                  "Train_Loss: {:>.8f} |".format(cost_train),
                                  "Valid_Loss: {:>.8f}".format(cost_valid))

                # Save Model
                # print('Saving model...')
                # saver = tf.train.Saver()
                # saver.save(sess, self.save_path + 'model.' + self.version + '.ckpt')

                # Prediction
                print('Predicting...')

                logits_ = sess.run(logits, {inputs: self.x_test,
                                            keep_prob: 1.0,
                                            is_train: False})

                logits_ = logits_.flatten()
                prob_test = 1.0 / (1.0 + np.exp(-logits_))

                prob_total.append(list(prob_test))

                utils.save_pred_to_csv(pred_path + 'dnn_cv_{}_'.format(cv_counter), self.id_test, prob_test)

            # Final Result
            print('======================================================================================================')
            print('Calculating final result...')

            prob_mean = np.mean(np.array(prob_total), axis=0)

            utils.save_pred_to_csv(pred_path + 'dnn_', self.id_test, prob_mean)


# Cross Validation

class CrossValidation:

    @staticmethod
    def sk_group_k_fold(x, y, e):

        era_k_fold = GroupKFold(n_splits=20)

        for train_index, valid_index in era_k_fold.split(x, y, e):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]

            yield x_train, y_train, x_valid, y_valid

    @staticmethod
    def sk_group_k_fold_with_weight(x, y, w, e):

        era_k_fold = GroupKFold(n_splits=20)

        for train_index, valid_index in era_k_fold.split(x, y, e):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]

            yield x_train, y_train, w_train, x_valid, y_valid, w_valid


# Grid Search

def grid_search(tr_x, tr_y, clf, params=None):

    era = tr_x[:, -1]
    np.delete(tr_x, 88, axis=1)

    #  grid_search = GridSearchCV(estimator=clf, param_grid=params, cv=group_k_fold(tr_x, tr_y))
    grid_search = GridSearchCV(estimator=clf, param_grid=params, scoring='neg_log_loss')

    # Start Grid Search
    print('Grid Seaching...')

    grid_search.fit(tr_x, tr_y, era)

    best_parameters = grid_search.best_estimator_.get_params()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")

    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == "__main__":

    # # HyperParameters
    # hyper_parameters = {'version': '1.0',
    #                     'epochs': 10,
    #                     'layers_number': 10,
    #                     'unit_number': [200, 400, 800, 800, 800, 800, 800, 800, 400, 200],
    #                     'learning_rate': 0.01,
    #                     'keep_probability': 0.75,
    #                     'batch_size': 512,
    #                     'display_step': 100,
    #                     'save_path': './checkpoints/',
    #                     'log_path': './log/'}
    #
    # pickled_data_path = './preprocessed_data/'
    #
    # tr, tr_y, tr_w, val_x, val_y, val_w = load_data(pickled_data_path)
    #
    # dnn = DNN(tr, tr_y, tr_w, val_x, val_y, val_w, hyper_parameters)
    # dnn.train()
    # print('Done!')

    pass