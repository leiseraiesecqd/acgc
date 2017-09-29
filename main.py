# import preprocess
import utils
import models
import stacking
import time
import os
from os.path import isdir
from sklearn.ensemble import ExtraTreesClassifier


preprocessed_data_path = './preprocessed_data/'
pred_path = './results/'
grid_search_log_path = './grid_search_logs/'
loss_log_path = './loss_logs/'
stack_output_path = './stacking_outputs/'


# Train single model
class TrainSingleModel:

    # Logistic Regression
    @staticmethod
    def lr_train():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        lr_parameters = {'C': 1.0,
                         'class_weight': None,
                         'dual': False,
                         'fit_intercept': True,
                         'intercept_scaling': 1,
                         'max_iter': 100,
                         # 'multi_class': 'multinomial',
                         'multi_class': 'ovr',
                         'n_jobs': -1,
                         'penalty': 'l2',
                         'solver': 'sag',
                         'tol': 0.0001,
                         'random_state': 1,
                         'verbose': 1,
                         'warm_start': False}

        LR = models.LRegression(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training Logistic Regression...')

        LR.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=lr_parameters)

    # Random Forest
    @staticmethod
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

        RF = models.RandomForest(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training Random Forest...')

        RF.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=rf_parameters)

    # Extra Trees
    @staticmethod
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

        ET = models.ExtraTrees(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training Extra Trees...')

        ET.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=et_parameters)

    # AdaBoost
    @staticmethod
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

        AB = models.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training AdaBoost...')

        AB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=ab_parameters)

    # GradientBoosting
    @staticmethod
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

        GB = models.GradientBoosting(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training GradientBoosting...')

        GB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=gb_parameters)

    # XGBoost
    @staticmethod
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

        XGB = models.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training XGBoost...')

        XGB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=xgb_parameters)

    @staticmethod
    def xgb_train_sklearn():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        xgb_parameters = {'max_depth': 3,
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'silent': True,
                          'objective': "binary:logistic",
                          #  'booster': 'gbtree',
                          #  'n_jobs':  1,
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
                          #  'random_state': 0,
                          'seed': None,
                          'missing': None}

        XGB = models.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training XGBoost...')

        XGB.train_sklearn(pred_path, loss_log_path, n_valid=4, n_cv=400, parameters=xgb_parameters)

    # LightGBM
    @staticmethod
    def lgb_train():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_train_g, x_test_g = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        lgb_parameters = {'application': 'binary',
                          'learning_rate': 0.002,
                          'num_leaves': 3,                # <2^(max_depth)
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

        LGBM = models.LightGBM(x_train, y_train, w_train, e_train, x_test, id_test, x_train_g, x_test_g)

        print('Start training LGBM...')

        LGBM.train(pred_path, loss_log_path, n_valid=4, n_cv=20, parameters=lgb_parameters)

    @staticmethod
    def lgb_train_sklearn():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_train_g, x_test_g = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        lgb_parameters = {'learning_rate': 0.006,
                          'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                          # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                          # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                          # 'boosting_type': 'rf',          # Random Forest.
                          'num_leaves': 3,                # <2^(max_depth)
                          'max_depth': 8,                 # default=-1
                          'n_estimators': 79,
                          'max_bin': 1005,
                          'subsample_for_bin': 1981,
                          'objective': 'binary',
                          'min_split_gain': 0.,
                          'min_child_weight': 1,
                          'min_child_samples': 0,
                          'subsample': 0.723,
                          'subsample_freq': 3,
                          'colsample_bytree': 0.11,
                          'reg_alpha': 0.,
                          'reg_lambda': 0.,
                          'silent': False}

        LGBM = models.LightGBM(x_train, y_train, w_train, e_train, x_test, id_test, x_train_g, x_test_g)

        print('Start training LGBM...')

        LGBM.train_sklearn(pred_path, loss_log_path, n_valid=4, n_cv=400, parameters=lgb_parameters)

    # DNN
    @staticmethod
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

        dnn = models.DeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, hyper_parameters)

        print('Start training DNN(TensorFlow)...')

        dnn.train(pred_path, n_valid=4, n_cv=20)

    # # DNN using Keras
    # @staticmethod
    # def dnn_keras_train():
    #
    #     x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
    #
    #     # HyperParameters
    #     hyper_parameters = {'epochs': 200,
    #                         'unit_number': [64, 32, 16, 8, 4, 1],
    #                         'learning_rate': 0.00001,
    #                         'keep_probability': 0.8,
    #                         'batch_size': 256}
    #
    #     dnn = models.KerasDeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, hyper_parameters)
    #
    #     print('Start training DNN(Keras)...')
    #
    #     dnn.train(pred_path, loss_log_path, n_valid=4, n_cv=20)


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

        LR = models.LRegression(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = LR.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'C': (0.2, 0.5, 1),
                           'max_iter': (50, 100, 200),
                           'tol': (0.001, 0.005, 0.01)
                           }

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                           params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Logistic Regression', parameters, parameters_grid)

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

        RF = models.RandomForest(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = RF.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           # 'n_estimators': (30, 31, 32),
                           'max_depth': (2, 3),
                           # 'max_features': (6, 7),
                           'min_samples_leaf': (286, 287),
                           'min_samples_split': (3972, 3974, 3976, 3978)
                           }

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                           params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Random Forest', parameters, parameters_grid)

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

        ET = models.ExtraTrees(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = ET.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'n_estimators': (30, 40, 50),
                           'max_depth': (5, 6),
                           'max_features': (6, 7),
                           'min_samples_leaf': (200, 250, 300),
                           'min_samples_split': (3000, 3500, 4000)
                           }

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                           params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Extra Trees', parameters, parameters_grid)

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

        AB = models.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = AB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'learning_rate': (0.002, 0.003, 0.005),
                           'n_estimators': (50, 100),
                           #  'random_state': 2,
                           #  'algorithm': 'SAMME.R',
                           #  'base_estimator': clf_et,
                           }

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                           params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('AdaBoost', parameters, parameters_grid)

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

        GB = models.GradientBoosting(x_train, y_train, w_train, e_train, x_test, id_test)

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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                           params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('GradientBoosting', parameters, parameters_grid)

    # XGBoost
    @staticmethod
    def xgb_grid_search():

        log_path = grid_search_log_path + 'xgb_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        parameters = {'objective': 'binary:logistic',
                      'learning_rate': 0.002,
                      'n_estimators': 100,
                      'max_depth': 9,
                      'min_child_weight': 5,
                      'max_delta_step': 0,
                      'silent': False,
                      'subsample': 0.8,
                      'colsample_bytree': 0.8,
                      'colsample_bylevel': 1,
                      'base_score': 0.5,
                      'gamma': 0,
                      'reg_alpha': 0,
                      'reg_lambda': 0,
                      'seed': 1
                      # 'missing': None,
                      # 'nthread': -1,
                      # 'scale_pos_weight': 1,
                      }

        XGB = models.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                           params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('XGBoost', parameters, parameters_grid)

    # LightGBM
    @staticmethod
    def lgb_grid_search():

        log_path = grid_search_log_path + 'lgb_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_train_g, x_test_g = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        parameters = {'learning_rate': 0.006,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 3,                # <2^(max_depth)
                      'max_depth': 8,                 # default=-1
                      'n_estimators': 79,
                      'max_bin': 1005,
                      'subsample_for_bin': 1981,
                      'objective': 'binary',
                      'min_split_gain': 0.,
                      'min_child_weight': 1,
                      'min_child_samples': 0,
                      'subsample': 0.723,
                      'subsample_freq': 3,
                      'colsample_bytree': 0.11,
                      'reg_alpha': 0.,
                      'reg_lambda': 0.,
                      'silent': False}

        LGB = models.LightGBM(x_train, y_train, w_train, e_train, x_test, id_test, x_train_g, x_test_g)

        clf = LGB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'learning_rate': (0.002, 0.005, 0.01),
                           'n_estimators': (30, 60, 90),
                           'num_leaves': (32, 64, 128),             # <2^(max_depth)
                           'colsample_bytree': (0.6, 0.8, 0.1),
                           'max_depth': (6, 8, 10),                 # default=-1
                           # 'min_data_in_leaf': 20,                  # default=20
                           # 'bagging_fraction': (0.5, 0.7, 0.9),
                           # 'feature_fraction': (0.5, 0.7, 0.9),
                           # 'subsample_for_bin': (50000, 100000, 150000),
                           # 'subsample_freq': (4, 6, 8),
                           # 'subsample': (0.6, 0.8, 1.0),
                           # 'max_bin': (255, 355, 455)
                           }

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20,
                           params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)


# Stacking

class ModelStacking:

    # Parameters for layer1
    @staticmethod
    def get_layer1_params():

        # Parameters of LightGBM
        lgb_params = {'learning_rate': 0.006,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      'num_leaves': 3,                # <2^(max_depth)
                      'max_depth': 8,                 # default=-1
                      'n_estimators': 79,
                      'max_bin': 1005,
                      'subsample_for_bin': 1981,
                      'objective': 'binary',
                      'min_split_gain': 0.,
                      'min_child_weight': 1,
                      'min_child_samples': 0,
                      'subsample': 0.723,
                      'subsample_freq': 3,
                      'colsample_bytree': 0.11,
                      'reg_alpha': 0.,
                      'reg_lambda': 0.,
                      'silent': False}

        # Parameters of XGBoost
        xgb_params = {'objective': 'binary:logistic',
                      'learning_rate': 0.002,
                      'n_estimators': 100,
                      'max_depth': 9,
                      'min_child_weight': 5,
                      'max_delta_step': 0,
                      'silent': False,
                      'subsample': 0.8,
                      'colsample_bytree': 0.8,
                      'colsample_bylevel': 1,
                      'gamma': 0,
                      'base_score': 0.5,
                      'reg_alpha': 0,
                      'reg_lambda': 0,
                      # 'missing': None,
                      # 'nthread': -1,
                      # 'scale_pos_weight': 1,
                      'seed': 1
                      }

        # Parameters of AdaBoost
        et_for_ab_params = {'bootstrap': True,
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
        clf_et_for_ab = ExtraTreesClassifier(**et_for_ab_params)
        ab_params = {'algorithm': 'SAMME.R',
                     'base_estimator': clf_et_for_ab,
                     'learning_rate': 0.0051,
                     'n_estimators': 9,
                     'random_state': 1}

        # Parameters of Random Forest
        rf_params = {'bootstrap': True,
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

        # Parameters of Extra Trees
        et_params = {'bootstrap': True,
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

        # Parameters of Gradient Boost
        gb_params = {'criterion': 'friedman_mse',
                     'init': None,
                     'learning_rate': 0.002,
                     'loss': 'deviance',
                     'max_depth': 5,
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

        # Parameters of Deep Neural Network
        dnn_params = {'version': '1.0',
                      'epochs': 10,
                      'unit_number': [48, 24, 12],
                      'learning_rate': 0.0001,
                      'keep_probability': 0.7,
                      'batch_size': 256,
                      'display_step': 100,
                      'save_path': './checkpoints/',
                      'log_path': './log/'}

        # List of parameters for layer1
        layer1_prams = [lgb_params,
                        # xgb_params,
                        # ab_params,
                        # rf_params,
                        # et_params,
                        # gb_params,
                        dnn_params
                        ]

        return layer1_prams

    # Parameters for layer2
    @staticmethod
    def get_layer2_params():

        # Parameters of LightGBM
        lgb_params = {'learning_rate': 0.002,
                      'boosting_type': 'gbdt',  # traditional Gradient Boosting Decision Tree.
                      'num_leaves': 32,         # <2^(max_depth)
                      'max_depth': 8,           # default=-1
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

        # Parameters of Deep Neural Network
        dnn_params = {'version': '1.0',
                      'epochs': 10,
                      'unit_number': [4, 2],
                      'learning_rate': 0.0001,
                      'keep_probability': 0.8,
                      'batch_size': 256,
                      'display_step': 100,
                      'save_path': './checkpoints/',
                      'log_path': './log/'}

        # List of parameters for layer1
        layer2_prams = [lgb_params,
                        dnn_params]

        return layer2_prams

    # Parameters for layer3
    @staticmethod
    def get_layer3_params():

        # Parameters of Deep Neural Network
        dnn_params = {'version': '1.0',
                      'epochs': 10,
                      'unit_number': [4, 2],
                      'learning_rate': 0.0001,
                      'keep_probability': 0.8,
                      'batch_size': 256,
                      'display_step': 100,
                      'save_path': './checkpoints/',
                      'log_path': './log/'}

        # List of parameters for layer1
        layer3_prams = [dnn_params]

        return layer3_prams

    @staticmethod
    def train():

        hyper_params = {'n_valid': (4, 4, 4),
                        'n_era': (20, 20, 20),
                        'n_epoch': (2, 2, 1)}

        layer1_prams = ModelStacking.get_layer1_params()
        layer2_prams = ModelStacking.get_layer2_params()
        layer3_prams = ModelStacking.get_layer3_params()

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_train_g, x_test_g = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        STK = stacking.DeepStack(x_train, y_train, w_train, e_train,
                                 x_test, id_test, x_train_g, x_test_g,
                                 pred_path, stack_output_path,
                                 hyper_params, layer1_prams, layer2_prams, layer3_prams)

        STK.stack()


if __name__ == "__main__":

    if not isdir(pred_path):
        os.makedirs(pred_path)

    if not isdir(pred_path + 'final_results/'):
        os.makedirs(pred_path + 'final_results/')

    if not isdir(grid_search_log_path):
        os.makedirs(grid_search_log_path)

    if not isdir(loss_log_path):
        os.makedirs(loss_log_path)

    if not isdir(stack_output_path):
        os.makedirs(stack_output_path)

    start_time = time.time()

    print('======================================================')
    print('Start training...')
    print('======================================================')

    # Logistic Regression
    TrainSingleModel.lr_train()

    # Random Forest
    # TrainSingleModel.rf_train()

    # Extra Trees
    # TrainSingleModel.et_train()

    # AdaBoost
    # TrainSingleModel.ab_train()

    # GradientBoosting
    # TrainSingleModel.gb_train()

    # XGBoost
    # TrainSingleModel.xgb_train()
    # TrainSingleModel.xgb_train_sklearn()

    # LightGBM
    # TrainSingleModel.lgb_train()
    # TrainSingleModel.lgb_train_sklearn()

    # DNN
    # TrainSingleModel.dnn_tf_train()
    # TrainSingleModel.dnn_keras_train()

    # Grid Search
    # GridSearch.lr_grid_search()
    # GridSearch.rf_grid_search()
    # GridSearch.et_grid_search()
    # GridSearch.ab_grid_search()
    # GridSearch.gb_grid_search()
    # GridSearch.xgb_grid_search()
    # GridSearch.lgb_grid_search()

    # Stacking
    # ModelStacking.train()

    print('======================================================')
    print('Done!')
    print('Time: {}s'.format(time.time() - start_time))
    print('======================================================')
