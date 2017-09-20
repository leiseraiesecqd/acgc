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
                     'min_impurity_decrease': 0.0,
                     'min_impurity_split': None,
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

    RF.train(pred_path, parameters=rf_parameters)


# AdaBoost

def ab_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    ab_parameters = {'algorithm': 'SAMME.R',
                     'base_estimator': None,
                     'learning_rate': 1.0,
                     'n_estimators': 50,
                     'random_state': 1}

    AB = model.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)

    AB.train(pred_path, parameters=ab_parameters)


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

    GB.train(pred_path, parameters=gb_parameters)


# XGBoost

def xgb_train():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    xgb_parameters = {'learning_rate': 0.05,
                      'n_estimators': 1000,
                      'max_depth': 10,
                      'min_child_weight': 5,
                      'gamma': 0,
                      'silent': 1,
                      'objective': 'binary:logistic',
                      'early_stopping_rounds': 50,
                      'subsample': 0.8,
                      'colsample_bytree': 0.8,
                      'eval_metric': 'logloss'}

    XGB = model.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

    XGB.train(pred_path, parameters=xgb_parameters)


# DNN

def dnn_train():
    # HyperParameters
    hyper_parameters = {'version': '1.0',
                        'epochs': 10,
                        'layers_number': 10,
                        'unit_number': [200, 400, 800, 800, 800, 800, 800, 800, 400, 200],
                        'learning_rate': 0.01,
                        'keep_probability': 0.75,
                        'batch_size': 512,
                        'display_step': 100,
                        'save_path': './checkpoints/',
                        'log_path': './log/'}

    print('Loading data set...')

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    dnn = model.DeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, hyper_parameters)

    dnn.train()


# Grid Search

def grid_search():

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

    xgb_parameters = {'base_score': 0.5,
                      'colsample_bylevel': 1,
                      'colsample_bytree': 0.8,
                      'gamma': 2,
                      'learning_rate': 0.05,
                      'max_delta_step': 0,
                      'max_depth': 3,
                      'min_child_weight': 1,
                      'missing': None,
                      'n_estimators': 100,
                      'nthread': -1,
                      'objective': 'binary:logistic',
                      'reg_alpha': 0,
                      'reg_lambda': 1,
                      'scale_pos_weight': 1,
                      'seed': 0,
                      'silent': 1,
                      'subsample': 0.8}

    XGB = model.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

    clf_xgb = XGB.clf(xgb_parameters)

    parameters_grid = None

    # parameters_grid = {'base_score': (0.4, 0.5, 0.6),
    #                    # 'colsample_bylevel': 1,
    #                    'colsample_bytree': (0.5, 0.8, 1.0),
    #                    # 'gamma': 2,
    #                    'learning_rate': (0.01, 0.05, 0.1, 0.15, 0.2),
    #                    # 'max_delta_step': 0,
    #                    'max_depth': (4, 6, 8, 10),
    #                    'min_child_weight': (0.8, 1.0, 1.2),
    #                    # 'missing': None,
    #                    # 'n_estimators': 100,
    #                    # 'nthread': -1,
    #                    # 'objective': 'binary:logistic',
    #                    # 'reg_alpha': 0,
    #                    # 'reg_lambda': 1,
    #                    # 'scale_pos_weight': 1,
    #                    # 'seed': 1,
    #                    # 'silent': True,
    #                    'subsample': (0.5, 0.8, 1.0)}

    model.grid_search(x_train, y_train, clf_xgb, parameters_grid)


if __name__ == "__main__":

    if not isdir(pred_path):
        os.mkdir(pred_path)

    start_time = time.time()

    #  rf_train()

    ab_train()

    gb_train()

    xgb_train()

    print('Done!')
    print('Using {:.3}s'.format(time.time() - start_time))
