# import preprocess
import utils
import model
import time
import os
from os.path import isdir


preprocessed_data_path = './preprocessed_data/'
pred_path = './result/'


# Random Forest

def rf_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    rf_parameters = {'bootstrap': True,
                     'class_weight': None,
                     'criterion': 'gini',
                     'max_depth': None,
                     'max_features': 'auto',
                     'max_leaf_nodes': None,
                     #  'min_impurity_decrease': 0.0,
                     #  'min_impurity_split': None,
                     'min_samples_leaf': 1,
                     'min_samples_split': 2,
                     'min_weight_fraction_leaf': 0.0,
                     'n_estimators': 50,
                     'n_jobs': 1,
                     'oob_score': False,
                     'random_state': 1,
                     'verbose': 1,
                     'warm_start': False}

    RF = model.RandomForest(x_train, y_train, w_train, e_train, x_test, id_test)

    RF.train(pred_path, n_valid=4, n_cv=20, parameters=rf_parameters)


# Extra Trees

def et_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    et_parameters = {'bootstrap': False,
                     'class_weight': None,
                     'criterion': 'gini',
                     'max_depth': 6,
                     'max_features': 'auto',
                     'max_leaf_nodes': None,
                     #  'min_impurity_decrease': 0.0,
                     'min_impurity_split': 0.1,
                     'min_samples_leaf': 1,
                     'min_samples_split': 2,
                     'min_weight_fraction_leaf': 0.0,
                     'n_estimators': 10,
                     'n_jobs': 1,
                     'oob_score': False,
                     'random_state': 1,
                     'verbose': 1,
                     'warm_start': False}

    ET = model.ExtraTrees(x_train, y_train, w_train, e_train, x_test, id_test)

    ET.train(pred_path, n_valid=4, n_cv=20, parameters=et_parameters)


# AdaBoost

def ab_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    ab_parameters = {'algorithm': 'SAMME.R',
                     'base_estimator': None,
                     'learning_rate': 1.0,
                     'n_estimators': 50,
                     'random_state': 1}

    AB = model.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)
    AB.train(pred_path, n_valid=4, n_cv=20, parameters=ab_parameters)


# GradientBoosting

def gb_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    gb_parameters = {'criterion': 'friedman_mse',
                     'init': None,
                     'learning_rate': 0.1,
                     'loss': 'deviance',
                     'max_depth': 3,
                     'max_features': None,
                     'max_leaf_nodes': None,
                     #  'min_impurity_decrease': 0.0,
                     'min_impurity_split': None,
                     'min_samples_leaf': 1,
                     'min_samples_split': 2,
                     'min_weight_fraction_leaf': 0.0,
                     'n_estimators': 50,
                     'presort': 'auto',
                     'random_state': 1,
                     'subsample': 1.0,
                     'verbose': 1,
                     'warm_start': False}

    GB = model.GradientBoosting(x_train, y_train, w_train, e_train, x_test, id_test)

    GB.train(pred_path, n_valid=4, n_cv=20, parameters=gb_parameters)


# XGBoost

def xgb_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    xgb_parameters = {'learning_rate': 0.05,
                      'n_estimators': 1000,
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

    XGB.train(pred_path, n_valid=4, n_cv=20, parameters=xgb_parameters)


# LightGBM

def lgb_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
    x_train_g, x_test_g = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

    lgb_parameters = {'application': 'binary',
                      'num_iterations': 100,          # this parameter is ignored, use num_boost_round input arguments of train and cv methods instead
                      'learning_rate': 0.01,
                      'num_leaves': 64,               # <2^(max_depth)
                      'tree_learner': 'serial',
                      'max_depth': 8,                 # default=-1
                      'min_data_in_leaf': 20,         # default=20
                      'feature_fraction': 0.7,        # default=1
                      'bagging_fraction': 0.8,        # default=1
                      'bagging_freq': 5,              # default=0 perform bagging every k iteration
                      'bagging_seed': 1,              # default=3
                      'early_stopping_rounds': 50,
                      'max_bin': 255,
                      'metric': 'binary_logloss',
                      'verbosity': 1}

    LGBM = model.LightGBM(x_train, y_train, w_train, e_train, x_test, id_test)

    LGBM.train(pred_path, n_valid=4, n_cv=20, x_train_g=x_train_g, x_test_g=x_test_g, parameters=lgb_parameters)


# DNN

def dnn_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    # HyperParameters
    hyper_parameters = {'version': '1.0',
                        'epochs': 200,
                        'layers_number': 6,
                        'unit_number': [64, 32, 16, 8, 4, 1],
                        'learning_rate': 0.00001,
                        'keep_probability': 0.7,
                        'batch_size': 256,
                        'display_step': 100,
                        'save_path': './checkpoints/',
                        'log_path': './log/'}

    print('Loading data set...')

    dnn = model.DeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, hyper_parameters)

    dnn.train(pred_path, n_valid=4, n_cv=20)


# Grid Search

def grid_search():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    xgb_parameters = {'learning_rate': 0.05,
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

    clf_xgb = XGB.clf(xgb_parameters)

    # parameters_grid = None

    parameters_grid = {'learning_rate': (0.01, 0.03, 0.05, 0.1, 0.2),
                       'n_estimators': (100, 200, 400),
                       'max_depth': (8, 10, 12),
                       #  'min_child_weight': 5,
                       #  'objective': 'binary:logistic',
                       #  'eval_metric': 'logloss',
                       #  'silent': 1,
                       #  'subsample': 0.8,
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

    model.grid_search(x_train, y_train, clf_xgb, parameters_grid)


if __name__ == "__main__":

    if not isdir(pred_path):
        os.makedirs(pred_path)

    start_time = time.time()

    # # Random Forest
    # print('Start training Random Forest...')
    # rf_train()

    # # Extra Trees
    # print('Start training Extra Trees...')
    # et_train()

    # # AdaBoost
    # print('Start training AdaBoost...')
    # ab_train()

    # # GradientBoosting
    # print('Start training GradientBoosting...')
    # gb_train()

    # # XGBoost
    # print('Start training XGBoost...')
    # xgb_train()

    #  LGBM
    print('Start training LGBM...')
    lgb_train()

    #  DNN
    # print('Start training DNN...')
    # dnn_train()

    # Grid Search
    #  grid_search()

    print('Done!')
    print('Using {:.3}s'.format(time.time() - start_time))
