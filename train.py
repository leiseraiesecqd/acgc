# import preprocess
import utils
import model
import time
import os
from os.path import isdir
from sklearn.ensemble import ExtraTreesClassifier


preprocessed_data_path = './preprocessed_data/'
pred_path = './results/'
grid_search_log_path = './grid_search_logs/'
loss_log_path = './loss_logs/'
importance_log_path = './importance_logs/'


# Logistic Regression

def lr_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    lr_parameters = {'C': 1.0,
                      'class_weight': None,
                      'dual': False,
                      'fit_intercept': 'True',
                      'intercept_scaling': 1,
                      'max_iter': 100,
                      'multi_class': 'multinomial',
                      'n_jobs': -1,
                      'penalty': 'l2',
                      'solver': 'sag',
                      'tol': 0.0001,
                      'random_state': 1,
                      'verbose': 2,
                      'warm_start': False}

    LR = model.LRegression(x_train, y_train, w_train, e_train, x_test, id_test)

    print('Start training Logistic Regression...')

    LR.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=lr_parameters)


# Random Forest

def rf_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    rf_parameters = {'bootstrap': True,
                     'class_weight': None,
                     'criterion': 'gini',
                     'max_depth': 2,
                     'max_features': 7,
                     'max_leaf_nodes': None,
                     'min_impurity_decrease': 0.0,
                     'min_samples_leaf': 286,
                     'min_samples_split': 3974,
                     'min_weight_fraction_leaf': 0.0,
                     'n_estimators': 32,
                     'n_jobs': -1,
                     'oob_score': True,
                     'random_state': 1,
                     'verbose': 2,
                     'warm_start': False}

    RF = model.RandomForest(x_train, y_train, w_train, e_train, x_test, id_test)

    print('Start training Random Forest...')

    RF.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=rf_parameters)


# Extra Trees

def et_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    et_parameters = {'bootstrap': True,
                     'class_weight': None,
                     'criterion': 'gini',
                     'max_depth': 2,
                     'max_features': 7,
                     'max_leaf_nodes': None,
                     'min_impurity_decrease': 0.0,
                     'min_samples_leaf': 357,
                     'min_samples_split': 4909,
                     'min_weight_fraction_leaf': 0.0,
                     'n_estimators': 20,
                     'n_jobs': -1,
                     'oob_score': True,
                     'random_state': 1,
                     'verbose': 2,
                     'warm_start': False}

    ET = model.ExtraTrees(x_train, y_train, w_train, e_train, x_test, id_test)

    print('Start training Extra Trees...')

    ET.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=et_parameters)


# AdaBoost

def ab_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    et_parameters = {'bootstrap': True,
                     'class_weight': None,
                     'criterion': 'gini',
                     'max_depth': 2,
                     'max_features': 7,
                     'max_leaf_nodes': None,
                     'min_impurity_decrease': 0.0,
                     'min_samples_leaf': 357,
                     'min_samples_split': 4909,
                     'min_weight_fraction_leaf': 0.0,
                     'n_estimators': 20,
                     'n_jobs': -1,
                     'oob_score': True,
                     'random_state': 1,
                     'verbose': 2,
                     'warm_start': False}

    clf_et = ExtraTreesClassifier(**et_parameters)

    ab_parameters = {'algorithm': 'SAMME.R',
                     'base_estimator': clf_et,
                     'learning_rate': 0.0051,
                     'n_estimators': 9,
                     'random_state': 1}

    AB = model.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)

    print('Start training AdaBoost...')

    AB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=ab_parameters)


# GradientBoosting

def gb_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    gb_parameters = {'criterion': 'friedman_mse',
                     'init': None,
                     'learning_rate': 0.05,
                     'loss': 'deviance',
                     'max_depth': 25,
                     'max_features': 'auto',
                     'max_leaf_nodes': None,
                     'min_impurity_decrease': 0.0,
                     'min_impurity_split': None,
                     'min_samples_leaf': 50,
                     'min_samples_split': 1000,
                     'min_weight_fraction_leaf': 0.0,
                     'n_estimators': 200,
                     'presort': 'auto',
                     'random_state': 1,
                     'subsample': 0.8,
                     'verbose': 2,
                     'warm_start': False}

    GB = model.GradientBoosting(x_train, y_train, w_train, e_train, x_test, id_test)

    print('Start training GradientBoosting...')

    GB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=gb_parameters)


# XGBoost

def xgb_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    xgb_parameters = {'learning_rate': 0.005,
                      'n_estimators': 200,
                      'gamma': 0,                       # 如果loss function小于设定值，停止产生子节点
                      'max_depth': 10,                  # default=6
                      'early_stopping_rounds': 50,
                      'min_child_weight': 5,            # default=1，建立每个模型所需最小样本数
                      'subsample': 0.8,                 # 建立树模型时抽取子样本占整个样本的比例
                      'colsample_bytree': 0.8,          # 建立树时对特征随机采样的比例
                      'objective': 'binary:logistic',
                      'eval_metric': 'logloss',
                      'seed': 1}

    XGB = model.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

    print('Start training XGBoost...')

    XGB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=xgb_parameters)


def xgb_train_sklearn():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    xgb_parameters = {'max_depth': 3,
                      'learning_rate': 0.1,
                      'n_estimators': 100,
                      'silent': True,
                      'objective': "binary:logistic",
                      'booster': 'gbtree',
                      'n_jobs':  1,
                      'nthread': None,
                      'gamma': 0,
                      'min_child_weight': 1,
                      'max_delta_step': 0,
                      'subsample': 1,
                      'colsample_bytree': 1,
                      'colsample_bylevel': 1,
                      'reg_alpha': 0,
                      'reg_lambda': 1,
                      'scale_pos_weight': 1,
                      'base_score': 0.5,
                      'random_state': 0,
                      'seed': None,
                      'missing': None}

    XGB = model.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

    print('Start training XGBoost...')

    XGB.train_sklearn(pred_path, loss_log_path, n_valid=4, n_cv=400, parameters=xgb_parameters)


# LightGBM

def lgb_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
    x_train_g, x_test_g = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

    lgb_parameters = {'application': 'binary',
                      'learning_rate': 0.002,
                      'num_leaves': 32,               # <2^(max_depth)
                      'tree_learner': 'serial',
                      'max_depth': 8,                 # default=-1
                      'min_data_in_leaf': 20,         # default=20
                      'feature_fraction': 0.5,        # default=1
                      'bagging_fraction': 0.6,        # default=1
                      'bagging_freq': 5,              # default=0 perform bagging every k iteration
                      'bagging_seed': 1,              # default=3
                      'early_stopping_rounds': 50,
                      'max_bin': 255,
                      'metric': 'binary_logloss',
                      'verbosity': 1}

    LGBM = model.LightGBM(x_train, y_train, w_train, e_train, x_test, id_test, x_train_g, x_test_g)

    print('Start training LGBM...')

    LGBM.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=lgb_parameters)


def lgb_train_sklearn():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
    x_train_g, x_test_g = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

    lgb_parameters = {'learning_rate': 0.002,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 32,               # <2^(max_depth)
                      'max_depth': 8,                 # default=-1
                      'n_estimators': 50,
                      'max_bin': 255,
                      'subsample_for_bin': 50000,
                      'objective': 'binary',
                      'min_split_gain': 0.,
                      'min_child_weight': 5,
                      'min_child_samples': 10,
                      'subsample': 0.6,
                      'subsample_freq': 5,
                      'colsample_bytree': 0.5,
                      'reg_alpha': 0.,
                      'reg_lambda': 0.,
                      'silent': False}

    LGBM = model.LightGBM(x_train, y_train, w_train, e_train, x_test, id_test, x_train_g, x_test_g)

    print('Start training LGBM...')

    LGBM.train_sklearn(pred_path, loss_log_path, n_valid=4, n_cv=400, parameters=lgb_parameters)


# DNN

def dnn_tf_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    # HyperParameters
    hyper_parameters = {'version': '1.0',
                        'epochs': 20,
                        'unit_number': [48, 24, 12, 6, 3],
                        'learning_rate': 0.0001,
                        'keep_probability': 0.8,
                        'batch_size': 256,
                        'display_step': 100,
                        'save_path': './checkpoints/',
                        'log_path': './log/'}

    print('Loading data set...')

    dnn = model.DeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, hyper_parameters)

    print('Start training DNN(TensorFlow)...')

    dnn.train(pred_path, n_valid=4, n_cv=20)


# DNN using Keras

def dnn_keras_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    # HyperParameters
    hyper_parameters = {'epochs': 200,
                        'unit_number': [64, 32, 16, 8, 4, 1],
                        'learning_rate': 0.00001,
                        'keep_probability': 0.8,
                        'batch_size': 256}

    dnn = model.KerasDeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, hyper_parameters)

    print('Start training DNN(Keras)...')

    dnn.train(pred_path, loss_log_path, n_valid=4, n_cv=20)


def print_grid_info(model_name, parameters, parameters_grid):

    print('\nModel: ' + model_name + '\n')
    print("Parameters:")
    print(parameters)
    print('\n')
    print("Parameters' grid:")
    print(parameters_grid)
    print('\n')


# Grid Search

class GridSearch:

    # LRegression:
    @staticmethod
    def lr_grid_search():

        log_path = grid_search_log_path + 'lr_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        parameters = {'C': 1.0,
                      'class_weight': None,
                      'dual': False,
                      'fit_intercept': 'True',
                      'intercept_scaling': 1,
                      'max_iter': 100,
                      'multi_class': 'multinomial',
                      'n_jobs': -1,
                      'penalty': 'l2',
                      'solver': 'sag',
                      'tol': 0.0001,
                      'random_state': 1,
                      'verbose': 2,
                      'warm_start': False}

        LR = model.LRegression(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = LR.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'C': (0.2, 0.5, 1),
                           'max_iter': (50, 100, 200),
                           'tol': (0.001, 0.005, 0.01)
                           }

        model.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                          params=parameters, params_grid=parameters_grid)

        print_grid_info('Logistic Regression', parameters, parameters_grid)

    # Random Forest
    @staticmethod
    def rf_grid_search():

        log_path = grid_search_log_path + 'rf_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        parameters = {'n_estimators': 32,
                      'bootstrap': True,
                      'class_weight': None,
                      'criterion': 'gini',
                      'max_depth': 6,
                      'max_features': 7,
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_samples_leaf': 300,
                      'min_samples_split': 4000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_jobs': -1,
                      'oob_score': True,
                      'random_state': 1,
                      'verbose': 2,
                      'warm_start': False}

        RF = model.RandomForest(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = RF.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           # 'n_estimators': (30, 31, 32),
                           'max_depth': (2, 3),
                           # 'max_features': (6, 7),
                           'min_samples_leaf': (286, 287),
                           'min_samples_split': (3972, 3974, 3976, 3978)
                           }

        model.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                          params=parameters, params_grid=parameters_grid)

        print_grid_info('Random Forest', parameters, parameters_grid)

    # Extra Trees
    @staticmethod
    def et_grid_search():

        log_path = grid_search_log_path + 'et_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(
            preprocessed_data_path)

        parameters = {'bootstrap': True,
                      'n_estimators': 50,
                      'class_weight': None,
                      'criterion': 'gini',
                      'max_depth': 25,
                      'max_features': 'auto',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_samples_leaf': 50,
                      'min_samples_split': 1000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_jobs': -1,
                      'oob_score': True,
                      'random_state': 1,
                      'verbose': 2,
                      'warm_start': False}

        ET = model.ExtraTrees(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = ET.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'n_estimators': (30, 40, 50),
                           'max_depth': (5, 6),
                           'max_features': (6, 7),
                           'min_samples_leaf': (200, 250, 300),
                           'min_samples_split': (3000, 3500, 4000)
                           }

        model.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                          params=parameters, params_grid=parameters_grid)

        print_grid_info('Extra Trees', parameters, parameters_grid)

    # AdaBoost
    @staticmethod
    def ab_grid_search():

        log_path = grid_search_log_path + 'ab_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        clf_et = ExtraTreesClassifier(n_estimators=100,
                                      max_depth=7,
                                      max_features=7,
                                      min_samples_leaf=500,
                                      min_samples_split=5000)

        parameters = {'algorithm': 'SAMME.R',
                      'base_estimator': clf_et,
                      'learning_rate': 0.005,
                      'n_estimators': 100,
                      'random_state': 1}

        AB = model.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = AB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'learning_rate': (0.002, 0.003, 0.005),
                           'n_estimators': (50, 100),
                           #  'random_state': 2,
                           #  'algorithm': 'SAMME.R',
                           #  'base_estimator': clf_et,
                           }

        model.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                          params=parameters, params_grid=parameters_grid)

        print_grid_info('AdaBoost', parameters, parameters_grid)

    # GradientBoosting
    @staticmethod
    def gb_grid_search():

        log_path = grid_search_log_path + 'gb_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        parameters = {'criterion': 'friedman_mse',
                      'init': None,
                      'learning_rate': 0.05,
                      'loss': 'deviance',
                      'max_depth': 25,
                      'max_features': 'auto',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_impurity_split': None,
                      'min_samples_leaf': 50,
                      'min_samples_split': 1000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_estimators': 200,
                      'presort': 'auto',
                      'random_state': 1,
                      'subsample': 0.8,
                      'verbose': 2,
                      'warm_start': False}

        GB = model.GradientBoosting(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = GB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'n_estimators': (20, 50, 100),
                           'learning_rate': (0.05, 0.2, 0.5),
                           'max_depth': (5, 10, 15),
                           'max_features': (6, 8, 10),
                           'min_samples_leaf': (300, 400, 500),
                           'min_samples_split': (3000, 4000, 5000),
                           'subsample': (0.6, 0.8, 1)
                           }

        model.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                          params=parameters, params_grid=parameters_grid)

        print_grid_info('GradientBoosting', parameters, parameters_grid)

    # XGBoost
    @staticmethod
    def xgb_grid_search():

        log_path = grid_search_log_path + 'xgb_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        parameters = {'learning_rate': 0.05,
                      'n_estimators': 1000,
                      'max_depth': 10,
                      'min_child_weight': 5,
                      'objective': 'binary:logistic',
                      #  'eval_metric': 'logloss',
                      'silent': 1,
                      'subsample': 0.8,
                      'colsample_bytree': 0.8,
                      'gamma': 0,
                      'base_score': 0.5,
                      # 'max_delta_step': 0,
                      # 'missing': None,
                      # 'nthread': -1,
                      # 'colsample_bylevel': 1,
                      # 'reg_alpha': 0,
                      # 'reg_lambda': 1,
                      # 'scale_pos_weight': 1,
                      'seed': 1}

        XGB = model.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = XGB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {'learning_rate': (0.002, 0.003, 0.005),
                           'n_estimators': (100, 200, 400, 800),
                           'max_depth': (8, 9, 10, 11),
                           #  'min_child_weight': 5,
                           #  'objective': 'binary:logistic',
                           #  'eval_metric': 'logloss',
                           #  'silent': 1,
                           'subsample': (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                           #  'colsample_bytree': 0.8,
                           #  'gamma': 0,
                           #  'base_score': 0.5,
                           # 'max_delta_step': 0,
                           # 'missing': None,
                           # 'nthread': -1,
                           # 'colsample_bylevel': 1,
                           # 'reg_alpha': 0,
                           # 'reg_lambda': 1,
                           # 'scale_pos_weight': 1,
                           #  'seed': 1
                           }

        model.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                          params=parameters, params_grid=parameters_grid)

        print_grid_info('XGBoost', parameters, parameters_grid)

    # LightGBM
    @staticmethod
    def lgb_grid_search():

        log_path = grid_search_log_path + 'lgb_'

        print('\nModel: LGBoost \n')

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_train_g, x_test_g = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        parameters = {'learning_rate': 0.002,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 32,               # <2^(max_depth)
                      'max_depth': 8,                 # default=-1
                      'n_estimators': 50,
                      'max_bin': 255,
                      'subsample_for_bin': 50000,
                      'objective': 'binary',
                      'min_split_gain': 0.,
                      'min_child_weight': 5,
                      'min_child_samples': 10,
                      'subsample': 0.6,
                      'subsample_freq': 5,
                      'colsample_bytree': 0.5,
                      'reg_alpha': 0.,
                      'reg_lambda': 0.,
                      'silent': False}

        print("Parameters:")
        print(parameters)
        print('\n')

        LGB = model.LightGBM(x_train, y_train, w_train, e_train, x_test, id_test, x_train_g, x_test_g)

        clf = LGB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'learning_rate': (0.002, 0.005, 0.01),
                           'n_estimators': (30, 60, 90),
                           'num_leaves': (32, 64, 128),               # <2^(max_depth)
                           'colsample_bytree': (0.6,0.8,0.1),
                           'max_depth': (6, 8, 10),                 # default=-1
                           # 'min_data_in_leaf': 20,         # default=20
                           # 'bagging_fraction': (0.5, 0.7, 0.9),
                           # 'feature_fraction': (0.5, 0.7, 0.9),
                           # 'subsample_for_bin': (50000, 100000, 150000),
                           # 'subsample_freq': (4, 6, 8),
                           # 'subsample': (0.6, 0.8, 1.0),
                           # 'max_bin': (255, 355, 455)
                           }

        print("Parameters' grid:")
        print(parameters_grid)
        print('\n')

        model.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                          params=parameters, params_grid=parameters_grid)

        print_grid_info('LightGBM', parameters, parameters_grid)


if __name__ == "__main__":

    if not isdir(pred_path):
        os.makedirs(pred_path)

    if not isdir(pred_path + 'final_results/'):
        os.makedirs(pred_path + 'final_results/')

    if not isdir(grid_search_log_path):
        os.makedirs(grid_search_log_path)

    if not isdir(loss_log_path):
        os.makedirs(loss_log_path)

    start_time = time.time()

    # Random Forest
    # rf_train()

    # Extra Trees
    # et_train()

    # AdaBoost
    # ab_train()

    # GradientBoosting
    # gb_train()

    # XGBoost
    # xgb_train()
    # xgb_train_sklearn()

    # LightGBM
    # lgb_train()
    # lgb_train_sklearn()

    # DNN
    # dnn_tf_train()
    # dnn_keras_train()

    # Grid Search
    # GridSearch.rf_grid_search()
    # GridSearch.ab_grid_search()
    # GridSearch.xgb_grid_search()
    GridSearch.lgb_grid_search()

    print('Done!')
    print('Using {:.3}s'.format(time.time() - start_time))
