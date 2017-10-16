import time
import utils
import models
import stacking
import prejudge
import preprocess
import numpy as np
import random
from cross_validation import CrossValidation


pred_path = './results/'
single_model_pred_path = pred_path + 'single_model/'
prejudge_pred_path = pred_path + 'prejudge/'
stack_pred_path = pred_path + 'stacking/'
auto_train_pred_path = pred_path + 'auto_train/'
log_path = './logs/'
csv_log_path = './logs/csv_logs/'
loss_log_path = log_path + 'loss_logs/'
prejudge_loss_log_path = loss_log_path + 'prejudge/'
dnn_log_path = log_path + 'dnn_logs/'
grid_search_log_path = log_path + 'grid_search_logs/'
data_path = './data/'
preprocessed_data_path = preprocess.preprocessed_path
prejudged_data_path = data_path + 'prejudged_data/'
stack_output_path = data_path + 'stacking_outputs/'
model_checkpoint_path = './checkpoints/'
dnn_checkpoint_path = model_checkpoint_path + 'dnn_checkpoints/'


path_list = [pred_path,
             single_model_pred_path,
             prejudge_pred_path,
             stack_pred_path,
             auto_train_pred_path,
             log_path,
             csv_log_path,
             grid_search_log_path,
             prejudge_loss_log_path,
             loss_log_path,
             dnn_log_path,
             data_path,
             prejudged_data_path,
             stack_output_path,
             model_checkpoint_path,
             dnn_checkpoint_path]


class TrainSingleModel:
    """
        Train single model
    """

    def __init__(self):
        pass

    @staticmethod
    def lr_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            Logistic Regression
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'C': 1.0,
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
                      'random_state': train_seed,
                      'verbose': 1,
                      'warm_start': False}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        LR = models.LRegression(x_train, y_train, w_train, e_train, x_test, id_test)

        LR.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                 n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                 cv_seed=cv_seed, parameters=parameters, show_importance=False,
                 show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def rf_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            Random Forest
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'bootstrap': True,
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
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        RF = models.RandomForest(x_train, y_train, w_train, e_train, x_test, id_test)

        RF.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                 n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                 cv_seed=cv_seed, parameters=parameters, show_importance=False,
                 show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def et_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            Extra Trees
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'bootstrap': True,
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
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        ET = models.ExtraTrees(x_train, y_train, w_train, e_train, x_test, id_test)

        ET.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                 n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                 cv_seed=cv_seed, parameters=parameters, show_importance=False,
                 show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def ab_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            AdaBoost
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

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
                         'random_state': train_seed,
                         'verbose': 2,
                         'warm_start': False}

        clf_et = models.ExtraTreesClassifier(**et_parameters)

        parameters = {'algorithm': 'SAMME.R',
                         'base_estimator': clf_et,
                         'learning_rate': 0.0051,
                         'n_estimators': 9,
                         'random_state': train_seed}
        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        AB = models.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        AB.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                 n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                 cv_seed=cv_seed, parameters=parameters, show_importance=False,
                 show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def gb_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            GradientBoosting
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

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
                      'random_state': train_seed,
                      'subsample': 0.8,
                      'verbose': 2,
                      'warm_start': False}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        GB = models.GradientBoosting(x_train, y_train, w_train, e_train, x_test, id_test)

        GB.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                 n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                 cv_seed=cv_seed, parameters=parameters, show_importance=False,
                 show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def xgb_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            XGBoost
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'eta': 0.003,
                      'gamma': 0,                       # 如果loss function小于设定值，停止产生子节点
                      'max_depth': 8,                   # default=6
                      'min_child_weight': 18,           # default=1，建立每个模型所需最小样本权重和
                      'subsample': 0.9,                 # 建立树模型时抽取子样本占整个样本的比例
                      'colsample_bytree': 0.7,          # 建立树时对特征随机采样的比例
                      'colsample_bylevel': 0.6,
                      'lambda': 0,
                      'alpha': 0,
                      'early_stopping_rounds': 30,
                      'nthread': -1,
                      'objective': 'binary:logistic',
                      'eval_metric': 'logloss',
                      'seed': train_seed}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        XGB = models.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test, num_boost_round=35)

        XGB.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                  n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                  cv_seed=cv_seed, parameters=parameters, show_importance=False,
                  show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def xgb_train_sklearn(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            XGBoost using scikit-learn module
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'max_depth': 3,
                      'learning_rate': 0.1,
                      'n_estimators': 100,
                      'silent': True,
                      'objective': "binary:logistic",
                      #  'booster': 'gbtree',
                      #  'n_jobs':  1,
                      'nthread': -1,
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
                      #  'random_state': train_seed,
                      'seed': train_seed,
                      'missing': None}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        XGB = models.SKLearnXGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        XGB.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                  n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                  cv_seed=cv_seed, parameters=parameters, show_importance=True,
                  show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def lgb_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            LightGBM
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'application': 'binary',
                      'boosting': 'gbdt',               # gdbt,rf,dart,goss
                      'learning_rate': 0.003,           # default=0.1
                      'num_leaves': 88,                 # default=31       <2^(max_depth)
                      'max_depth': 10,                   # default=-1
                      'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,            # default=1
                      'feature_fraction_seed': 10,      # default=2
                      'bagging_fraction': 0.8,          # default=1
                      'bagging_freq': 1,                # default=0        perform bagging every k iteration
                      'bagging_seed': 19,               # default=3
                      'lambda_l1': 0,                   # default=0
                      'lambda_l2': 0,                   # default=0
                      'min_gain_to_split': 0,           # default=0
                      'max_bin': 225,                  # default=255
                      'min_data_in_bin': 5,             # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,      # default=0
                      'seed': train_seed}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        LGBM = models.LightGBM(x_g_train, y_train, w_train, e_train, x_g_test, id_test, num_boost_round=65)

        # cv_generator = CrossValidation.era_k_fold_with_weight_all_random
        # cv_generator = CrossValidation.random_split_with_weight
        cv_generator = CrossValidation.era_k_fold_with_weight_balance

        LGBM.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                   n_valid=4, n_cv=20, n_era=20, train_seed=train_seed, auto_train_pred_path=auto_train_path,
                   cv_seed=cv_seed, parameters=parameters, show_importance=False, save_cv_pred=False,
                   show_accuracy=True, save_csv_log=True, csv_idx=idx, cv_generator=cv_generator)

    @staticmethod
    def lgb_train_sklearn(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            LightGBM using scikit-learn module
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'learning_rate': 0.003,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 80,               # <2^(max_depth)
                      'max_depth': 7,                 # default=-1
                      'n_estimators': 50,
                      'max_bin': 1005,
                      'subsample_for_bin': 1981,
                      'objective': 'binary',
                      'min_split_gain': 0.,
                      'min_child_weight': 1,
                      'min_child_samples': 0,
                      'subsample': 0.8,
                      'subsample_freq': 5,
                      'colsample_bytree': 0.8,
                      'reg_alpha': 0.5,
                      'reg_lambda': 0.5,
                      'silent': False,
                      'seed': train_seed}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        LGBM = models.SKLearnLightGBM(x_g_train, y_train, w_train, e_train, x_g_test, id_test)

        LGBM.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                   n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                   cv_seed=cv_seed, parameters=parameters, show_importance=False,
                   show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def cb_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            CatBoost
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'iterations': 50,
                      'learning_rate': 0.003,
                      'depth': 8,                            # Depth of the tree.
                      'l2_leaf_reg': 0.3,                      # L2 regularization coefficient.
                      'rsm': 1,                              # The percentage of features to use at each iteration.
                      'bagging_temperature': 0.9,              # Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
                      'loss_function': 'Logloss',
                      'border': 0.5,
                      'border_count': 128,
                      'feature_border_type': 'MinEntropy',
                      'fold_permutation_block_size': 1,
                      'od_pval': None,                       # Use overfitting detector to stop training when reaching a specified threshold.
                      'od_wait': None,                       # Number of iterations which overfitting detector will wait after new best error.
                      'od_type': 'IncToDec',                 # Type of overfitting detector which will be used in program.
                      'gradient_iterations': None,           # The number of gradient steps when calculating the values in leaves.
                      'leaf_estimation_method': 'Gradient',  # The method used to calculate the values in leaves.
                      'thread_count': None,                  # Number of parallel threads used to run CatBoost.
                      'random_seed': train_seed,
                      'use_best_model': False,               # To limit the number of trees in predict() using information about the optimal value of the error function.
                      'verbose': True,
                      'ctr_description': None,               # Binarization settings for categorical features.
                      'ctr_border_count': 16,                # The number of partitions for Categ features.
                      'ctr_leaf_count_limit': None,          # The maximum number of leafs with categorical features.
                      'priors': None,                        # Use priors when training.
                      'has_time': False,                     # To use the order in which objects are represented in the input data.
                      'name': 'experiment',
                      'ignored_features': None,
                      'train_dir': None,
                      'custom_loss': None,
                      'eval_metric': 'Logloss',
                      'class_weights': None}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        CB = models.CatBoost(x_g_train, y_train, w_train, e_train, x_g_test, id_test)

        CB.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                 n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                 cv_seed=cv_seed, parameters=parameters, show_importance=True,
                 show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    @staticmethod
    def dnn_tf_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            Deep Neural Networks
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        # HyperParameters
        parameters = {'version': '1.0',
                      'epochs': 100,
                      'unit_number': [48, 24, 12],
                      'learning_rate': 0.01,
                      'keep_probability': 1.0,
                      'batch_size': 128,
                      'seed': train_seed,
                      'display_step': 100,
                      'save_path': dnn_checkpoint_path,
                      'log_path': dnn_log_path}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        print('Loading data set...')

        dnn = models.DeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, parameters)

        dnn.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'single_',
                  n_valid=4, n_cv=20, n_era=20, train_seed=train_seed, cv_seed=cv_seed, show_accuracy=True,
                  save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)

    # # DNN using Keras
    # @staticmethod
    # def dnn_keras_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
    #
    #     x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
    #
    #     # HyperParameters
    #     hyper_parameters = {'epochs': 200,
    #                         'unit_number': [64, 32, 16, 8, 4, 1],
    #                         'learning_rate': 0.00001,
    #                         'keep_probability': 0.8,
    #                         'batch_size': 256}

        # # Grid Search
        # if grid_search_tuple is not None:
        #     parameters[grid_search_tuple[0]] = grid_search_tuple[1]
    #
    #     dnn = models.KerasDeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, hyper_parameters)
    #
    #     dnn.train(pred_path, loss_log_path, n_valid=4, n_cv=20, cv_seed=cv_seed)

    @staticmethod
    def stack_lgb_train(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None, auto_idx=None):
        """
            LightGBM for stack layer
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)
        blender_x_tree, blender_test_tree, blender_x_g_tree, blender_test_g_tree\
            = utils.load_stacked_data(stack_output_path + 'auto_{}_l2_'.format(auto_idx))

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        g_train = x_g_train[:, -1]
        g_test = x_g_test[:, -1]

        x_train_reuse = x_train[:, :87]
        x_test_reuse = x_test[:, :87]

        print('------------------------------------------------------')
        print('Stacking Reused Features of Train Set...')
        # n_sample * (n_feature + n_reuse)
        blender_x_tree = np.concatenate((blender_x_tree, x_train_reuse), axis=1)
        # n_sample * (n_feature + n_reuse + 1)
        blender_x_g_tree = np.column_stack((blender_x_tree, g_train))

        print('------------------------------------------------------')
        print('Stacking Reused Features of Test Set...')
        # n_sample * (n_feature + n_reuse)
        blender_test_tree = np.concatenate((blender_test_tree, x_test_reuse), axis=1)
        # n_sample * (n_feature + n_reuse + 1)
        blender_test_g_tree = np.column_stack((blender_test_tree, g_test))

        parameters = {'application': 'binary',
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.002,               # default=0.1
                      'num_leaves': 80,                     # default=31       <2^(max_depth)
                      'max_depth': 7,                       # default=-1
                      'min_data_in_leaf': 2000,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.8,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.6,              # default=1
                      'bagging_freq': 5,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 2250,                      # default=255
                      'min_data_in_bin': 5,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,          # default=0
                      'seed': train_seed}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        LGB = models.LightGBM(blender_x_g_tree, y_train, w_train, e_train,
                              blender_test_g_tree, id_test, num_boost_round=65)

        # cv_generator = CrossValidation.era_k_fold_with_weight_balance

        LGB.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'stack_final_',
                  n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                  cv_seed=cv_seed, parameters=parameters, show_importance=False,
                  show_accuracy=True, save_csv_log=True, csv_idx=idx, auto_train_pred_path=auto_train_path)


class ChampionModel:

    def __init__(self):
        pass

    @staticmethod
    def Christ1991(train_seed, cv_seed, save_auto_train_results=False, idx=None, grid_search_tuple=None):
        """
            Model of week3 champion
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        parameters = {'application': 'binary',
                      'learning_rate': 0.002,
                      'num_leaves': 128,              # <2^(max_depth)
                      'tree_learner': 'serial',
                      'max_depth': 8,                 # default=-1
                      'min_data_in_leaf': 20,         # default=20
                      'feature_fraction': 0.8,        # default=1
                      'bagging_fraction': 0.8,        # default=1
                      'bagging_freq': 5,              # default=0 perform bagging every k iteration
                      'bagging_seed': 6,              # default=3
                      'early_stopping_rounds': 50,
                      'max_bin': 255,
                      'metric': 'binary_logloss',
                      'verbosity': 1}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        LGBM = models.LightGBM(x_g_train, y_train, w_train, e_train, x_g_test, id_test, num_boost_round=65)

        # cv_generator = CrossValidation.era_k_fold_with_weight_all_random

        LGBM.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'christ1991_',
                   n_valid=4, n_cv=20, n_era=20, train_seed=train_seed,
                   cv_seed=cv_seed, parameters=parameters, show_importance=False, show_accuracy=False,
                   save_csv_log=True, csv_idx=idx, cv_generator=None, auto_train_pred_path=auto_train_path)


class GridSearch:
    """
        Grid Search
    """

    def __init__(self):
        pass

    @staticmethod
    def lr_grid_search(train_seed, cv_seed):
        """
            Logistic Regression
        """

        _log_path = grid_search_log_path + 'lr_'

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
                      'random_state': train_seed,
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

        models.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Logistic Regression', parameters, parameters_grid)

    @staticmethod
    def rf_grid_search(train_seed, cv_seed):
        """
            Random Forest
        """

        _log_path = grid_search_log_path + 'rf_'

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
                      'random_state': train_seed,
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

        models.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Random Forest', parameters, parameters_grid)

    @staticmethod
    def et_grid_search(train_seed, cv_seed):
        """
            Extra Trees
        """

        _log_path = grid_search_log_path + 'et_'

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
                      'random_state': train_seed,
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

        models.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Extra Trees', parameters, parameters_grid)

    @staticmethod
    def ab_grid_search(train_seed, cv_seed):
        """
            AdaBoost
        """

        _log_path = grid_search_log_path + 'ab_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

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
                            'random_state': train_seed,
                            'verbose': 2,
                            'warm_start': False}
        clf_et_for_ab = models.ExtraTreesClassifier(**et_for_ab_params)

        parameters = {'algorithm': 'SAMME.R',
                      'base_estimator': clf_et_for_ab,
                      'learning_rate': 0.005,
                      'n_estimators': 100,
                      'random_state': train_seed}

        AB = models.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = AB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'learning_rate': (0.002, 0.003, 0.005),
                           'n_estimators': (50, 100),
                           #  'algorithm': 'SAMME.R',
                           #  'base_estimator': clf_et,
                           }

        models.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('AdaBoost', parameters, parameters_grid)

    @staticmethod
    def gb_grid_search(train_seed, cv_seed):
        """
            GradientBoosting
        """

        _log_path = grid_search_log_path + 'gb_'

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
                      'random_state': train_seed,
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

        models.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('GradientBoosting', parameters, parameters_grid)

    @staticmethod
    def xgb_grid_search(train_seed, cv_seed):
        """
            XGBoost
        """

        _log_path = grid_search_log_path + 'xgb_'

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
                      'nthread': -1,
                      'seed': train_seed
                      # 'missing': None,
                      # 'nthread': -1,
                      # 'scale_pos_weight': 1,
                      }

        XGB = models.SKLearnXGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        clf = XGB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {'learning_rate': (0.002, 0.005, 0.01),
                           'n_estimators': (20, 50, 100, 150),
                           'max_depth': (5, 7, 9),
                           # 'subsample': 0.8,
                           # 'colsample_bytree': 0.8,
                           # 'colsample_bylevel': 1,
                           # 'gamma': 0,
                           # 'min_child_weight': 1,
                           # 'max_delta_step': 0,
                           # 'base_score': 0.5,
                           # 'reg_alpha': 0,
                           # 'reg_lambda': 0,
                           }

        models.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('XGBoost', parameters, parameters_grid)

    @staticmethod
    def lgb_grid_search(train_seed, cv_seed):
        """
            LightGBM
        """

        _log_path = grid_search_log_path + 'lgb_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

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
                      'silent': False,
                      'seed': train_seed}

        LGB = models.SKLearnLightGBM(x_g_train, y_train, w_train, e_train, x_g_test, id_test)

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

        models.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)

    # Stacking Layer LightGBM
    @staticmethod
    def stack_lgb_grid_search(train_seed, cv_seed):

        _log_path = grid_search_log_path + 'stk_lgb_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)
        blender_x_tree, blender_test_tree, blender_x_g_tree, blender_test_g_tree \
            = utils.load_stacked_data(stack_output_path + 'l2_')

        g_train = x_g_train[:, -1]
        x_train_reuse = x_train[:, :88]

        print('------------------------------------------------------')
        print('Stacking Reused Features of Train Set...')
        blender_x_tree = np.concatenate((blender_x_tree, x_train_reuse), axis=1)
        blender_x_g_tree = np.column_stack((blender_x_tree, g_train))

        parameters = {'learning_rate': 0.006,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 3,  # <2^(max_depth)
                      'max_depth': 8,  # default=-1
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
                      'silent': False,
                      'random_state': train_seed}

        LGB = models.SKLearnLightGBM(blender_x_g_tree, y_train, w_train, e_train, blender_test_g_tree, id_test)

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

        models.grid_search(_log_path, blender_x_tree, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)


class PrejudgeTraining:
    """
        Prejudge - Training by Split Era sign
    """

    def __init__(self):
        pass

    @staticmethod
    def get_binary_models_parameters(train_seed):
        """
            Set Parameters for models of PrejudgeBinary
        """

        era_training_params = {'application': 'binary',
                               'boosting': 'gbdt',               # gdbt,rf,dart,goss
                               'learning_rate': 0.1,             # default=0.1
                               'num_leaves': 88,                 # default=31       <2^(max_depth)
                               'max_depth': 10,                  # default=-1
                               'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                               'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3     reduce over-fit
                               'feature_fraction': 0.5,          # default=1
                               'feature_fraction_seed': 10,      # default=2
                               'bagging_fraction': 0.8,          # default=1
                               'bagging_freq': 1,                # default=0        perform bagging every k iteration
                               'bagging_seed': 19,               # default=3
                               'lambda_l1': 5,                   # default=0
                               'lambda_l2': 5,                   # default=0
                               'min_gain_to_split': 0,           # default=0
                               'max_bin': 225,                   # default=255
                               'min_data_in_bin': 5,             # default=5
                               'metric': 'binary_logloss',
                               'num_threads': -1,
                               'verbosity': 1,
                               'early_stopping_rounds': 50,      # default=0
                               'seed': train_seed}

        positive_params = {'application': 'binary',
                           'boosting': 'gbdt',               # gdbt,rf,dart,goss
                           'learning_rate': 0.003,           # default=0.1
                           'num_leaves': 88,                 # default=31       <2^(max_depth)
                           'max_depth': 10,                  # default=-1
                           'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                           'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3     reduce over-fit
                           'feature_fraction': 0.5,          # default=1
                           'feature_fraction_seed': 10,      # default=2
                           'bagging_fraction': 0.8,          # default=1
                           'bagging_freq': 1,                # default=0        perform bagging every k iteration
                           'bagging_seed': 19,               # default=3
                           'lambda_l1': 0,                   # default=0
                           'lambda_l2': 0,                   # default=0
                           'min_gain_to_split': 0,           # default=0
                           'max_bin': 225,                   # default=255
                           'min_data_in_bin': 5,             # default=5
                           'metric': 'binary_logloss',
                           'num_threads': -1,
                           'verbosity': 1,
                           'early_stopping_rounds': 50,      # default=0
                           'seed': train_seed}

        negative_params = {'application': 'binary',
                           'boosting': 'gbdt',               # gdbt,rf,dart,goss
                           'learning_rate': 0.003,           # default=0.1
                           'num_leaves': 88,                 # default=31       <2^(max_depth)
                           'max_depth': 10,                  # default=-1
                           'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                           'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3     reduce over-fit
                           'feature_fraction': 0.5,          # default=1
                           'feature_fraction_seed': 10,      # default=2
                           'bagging_fraction': 0.8,          # default=1
                           'bagging_freq': 1,                # default=0        perform bagging every k iteration
                           'bagging_seed': 19,               # default=3
                           'lambda_l1': 0,                   # default=0
                           'lambda_l2': 0,                   # default=0
                           'min_gain_to_split': 0,           # default=0
                           'max_bin': 225,                   # default=255
                           'min_data_in_bin': 5,             # default=5
                           'metric': 'binary_logloss',
                           'num_threads': -1,
                           'verbosity': 1,
                           'early_stopping_rounds': 50,      # default=0
                           'seed': train_seed}

        models_parameters = [era_training_params, positive_params, negative_params]

        return models_parameters

    @staticmethod
    def get_multiclass_models_parameters(train_seed):
        """
            Set Parameters for models of PrejudgeMultiClass
        """

        era_training_params = {'application': 'multiclass',
                               'num_class': 20,
                               'learning_rate': 0.2,
                               'num_leaves': 80,            # <2^(max_depth)
                               'tree_learner': 'serial',
                               'max_depth': 7,              # default=-1
                               'min_data_in_leaf': 2000,    # default=20
                               'feature_fraction': 0.5,     # default=1
                               'bagging_fraction': 0.6,     # default=1
                               'bagging_freq': 5,           # default=0 perform bagging every k iteration
                               'bagging_seed': 1,           # default=3
                               'early_stopping_rounds': 50,
                               'max_bin': 50,
                               'metric': 'multi_logloss',
                               'verbosity': 0,
                               'seed': train_seed}

        multiclass_params = {'application': 'binary',
                             'learning_rate': 0.002,
                             'num_leaves': 80,              # <2^(max_depth)
                             'tree_learner': 'serial',
                             'max_depth': 7,                # default=-1
                             'min_data_in_leaf': 2000,      # default=20
                             'feature_fraction': 0.5,       # default=1
                             'bagging_fraction': 0.6,       # default=1
                             'bagging_freq': 5,             # default=0 perform bagging every k iteration
                             'bagging_seed': 1,             # default=3
                             'early_stopping_rounds': 65,
                             'max_bin': 50,
                             'metric': 'binary_logloss',
                             'verbosity': 1,
                             'seed': train_seed}

        models_parameters = [era_training_params, multiclass_params]

        return models_parameters

    @staticmethod
    def binary_train(train_seed, cv_seed):
        """
            Training model of PrejudgeBinary
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)
        x_train_p, y_train_p, w_train_p, e_train_p, x_g_train_p \
            = utils.load_preprocessed_positive_pd_data(preprocessed_data_path)
        x_train_n, y_train_n, w_train_n, e_train_n, x_g_train_n \
            = utils.load_preprocessed_negative_pd_data(preprocessed_data_path)

        models_parameters = PrejudgeTraining.get_binary_models_parameters(train_seed)

        positive_era_list = preprocess.positive_era_list
        negative_era_list = preprocess.negative_era_list

        hyper_parameters = {'cv_seed': cv_seed,
                            'train_seed': train_seed,
                            'n_splits_e': 10,
                            'num_boost_round_e': 2000,
                            'n_cv_e': 10,
                            'n_valid_p': 2,
                            'n_cv_p': 18,
                            'n_era_p': len(positive_era_list),
                            'num_boost_round_p': 65,
                            'era_list_p': positive_era_list,
                            'n_valid_n': 1,
                            'n_cv_n': 8,
                            'n_era_n': len(negative_era_list),
                            'num_boost_round_n': 65,
                            'era_list_n': negative_era_list,
                            'force_convert_era': True,
                            'use_weight': True,
                            'show_importance': False,
                            'show_accuracy': True}

        PES = prejudge.PrejudgeBinary(x_train, y_train, w_train, e_train, x_g_train, x_train_p,
                                      y_train_p, w_train_p, e_train_p, x_g_train_p, x_train_n, y_train_n,
                                      w_train_n, e_train_n, x_g_train_n, x_test, id_test, x_g_test,
                                      pred_path=prejudge_pred_path, prejudged_data_path=prejudged_data_path,
                                      loss_log_path=prejudge_loss_log_path, csv_log_path=csv_log_path + 'prejudge_',
                                      models_parameters=models_parameters, hyper_parameters=hyper_parameters)

        PES.train(load_pickle=False, load_pickle_path=None)

    @staticmethod
    def multiclass_train(train_seed, cv_seed):
        """
            Training model of PrejudgeMultiClass
        """

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        models_parameters = PrejudgeTraining.get_multiclass_models_parameters(train_seed)

        hyper_parameters = {'cv_seed': cv_seed,
                            'train_seed': train_seed,
                            'n_splits_e': 10,
                            'num_boost_round_e': 500,
                            'n_cv_e': 10,
                            'n_era': 20,
                            'n_valid_m': 4,
                            'n_cv_m': 20,
                            'num_boost_round_m': 65,
                            'use_weight': True,
                            'show_importance': False,
                            'show_accuracy': True}

        PES = prejudge.PrejudgeMultiClass(x_train, y_train, w_train, e_train, x_g_train, x_test, id_test, x_g_test,
                                          pred_path=prejudge_pred_path, prejudged_data_path=prejudged_data_path,
                                          loss_log_path=prejudge_loss_log_path, csv_log_path=csv_log_path + 'prejudge_',
                                          models_parameters=models_parameters, hyper_parameters=hyper_parameters)

        PES.train(load_pickle=True, load_pickle_path=None)


class ModelStacking:
    """
        Stacking
    """

    def __init__(self):
        pass

    @staticmethod
    def get_layer1_params(train_seed):
        """
            Set Parameters for models of layer1
        """

        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',               # gdbt,rf,dart,goss
                      'learning_rate': 0.003,           # default=0.1
                      'num_leaves': 88,                 # default=31       <2^(max_depth)
                      'max_depth': 10,                   # default=-1
                      'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,            # default=1
                      'feature_fraction_seed': 10,      # default=2
                      'bagging_fraction': 0.8,          # default=1
                      'bagging_freq': 1,                # default=0        perform bagging every k iteration
                      'bagging_seed': 19,               # default=3
                      'lambda_l1': 0,                   # default=0
                      'lambda_l2': 0,                   # default=0
                      'min_gain_to_split': 0,           # default=0
                      'max_bin': 225,                  # default=255
                      'min_data_in_bin': 5,             # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,      # default=0
                      'seed': train_seed}

        # Parameters of XGBoost
        xgb_params = {'eta': 0.003,
                      'gamma': 0,                       # 如果loss function小于设定值，停止产生子节点
                      'max_depth': 8,                   # default=6
                      'min_child_weight': 18,           # default=1，建立每个模型所需最小样本权重和
                      'subsample': 0.9,                 # 建立树模型时抽取子样本占整个样本的比例
                      'colsample_bytree': 0.7,          # 建立树时对特征随机采样的比例
                      'colsample_bylevel': 0.6,
                      'lambda': 0,
                      'alpha': 0,
                      'early_stopping_rounds': 30,
                      'nthread': -1,
                      'objective': 'binary:logistic',
                      'eval_metric': 'logloss',
                      'seed': train_seed}

        # # Parameters of XGBoost
        # xgb_params = {'eta': 0.008,
        #               'gamma': 0,                           # 如果loss function小于设定值，停止产生子节点
        #               'max_depth': 7,                       # default=6
        #               'min_child_weight': 15,               # default=1，建立每个模型所需最小样本权重和
        #               'subsample': 0.8,                     # 建立树模型时抽取子样本占整个样本的比例
        #               'colsample_bytree': 0.7,              # 建立树时对特征随机采样的比例
        #               'colsample_bylevel': 0.6,
        #               'lambda': 2500,
        #               'alpha': 0,
        #               'early_stopping_rounds': 30,
        #               'nthread': -1,
        #               'objective': 'binary:logistic',
        #               'eval_metric': 'logloss',
        #               'seed': train_seed}

        # # Parameters of AdaBoost
        # et_for_ab_params = {'bootstrap': True,
        #                     'class_weight': None,
        #                     'criterion': 'gini',
        #                     'max_depth': 2,
        #                     'max_features': 7,
        #                     'max_leaf_nodes': None,
        #                     'min_impurity_decrease': 0.0,
        #                     'min_samples_leaf': 357,
        #                     'min_samples_split': 4909,
        #                     'min_weight_fraction_leaf': 0.0,
        #                     'n_estimators': 20,
        #                     'n_jobs': -1,
        #                     'oob_score': True,
        #                     'random_state': train_seed,
        #                     'verbose': 2,
        #                     'warm_start': False}
        # clf_et_for_ab = models.ExtraTreesClassifier(**et_for_ab_params)
        # ab_params = {'algorithm': 'SAMME.R',
        #              'base_estimator': clf_et_for_ab,
        #              'learning_rate': 0.0051,
        #              'n_estimators': 9,
        #              'random_state': train_seed}
        #
        # # Parameters of Random Forest
        # rf_params = {'bootstrap': True,
        #              'class_weight': None,
        #              'criterion': 'gini',
        #              'max_depth': 2,
        #              'max_features': 7,
        #              'max_leaf_nodes': None,
        #              'min_impurity_decrease': 0.0,
        #              'min_samples_leaf': 286,
        #              'min_samples_split': 3974,
        #              'min_weight_fraction_leaf': 0.0,
        #              'n_estimators': 32,
        #              'n_jobs': -1,
        #              'oob_score': True,
        #              'random_state': train_seed,
        #              'verbose': 2,
        #              'warm_start': False}
        #
        # # Parameters of Extra Trees
        # et_params = {'bootstrap': True,
        #              'class_weight': None,
        #              'criterion': 'gini',
        #              'max_depth': 2,
        #              'max_features': 7,
        #              'max_leaf_nodes': None,
        #              'min_impurity_decrease': 0.0,
        #              'min_samples_leaf': 357,
        #              'min_samples_split': 4909,
        #              'min_weight_fraction_leaf': 0.0,
        #              'n_estimators': 20,
        #              'n_jobs': -1,
        #              'oob_score': True,
        #              'random_state': train_seed,
        #              'verbose': 2,
        #              'warm_start': False}
        #
        # # Parameters of Gradient Boost
        # gb_params = {'criterion': 'friedman_mse',
        #              'init': None,
        #              'learning_rate': 0.002,
        #              'loss': 'deviance',
        #              'max_depth': 5,
        #              'max_features': 'auto',
        #              'max_leaf_nodes': None,
        #              'min_impurity_decrease': 0.0,
        #              'min_impurity_split': None,
        #              'min_samples_leaf': 50,
        #              'min_samples_split': 1000,
        #              'min_weight_fraction_leaf': 0.0,
        #              'n_estimators': 200,
        #              'presort': 'auto',
        #              'random_state': train_seed,
        #              'subsample': 0.8,
        #              'verbose': 2,
        #              'warm_start': False}

        # Parameters of Deep Neural Network
        dnn_params = {'version': '1.0',
                      'epochs': 5,
                      'unit_number': [48, 24, 12],
                      'learning_rate': 0.0001,
                      'keep_probability': 0.4,
                      'batch_size': 256,
                      'seed': train_seed,
                      'display_step': 100,
                      'save_path': dnn_checkpoint_path,
                      'log_path': dnn_log_path}

        # List of parameters for layer1
        layer1_params = [
                        lgb_params,
                        xgb_params,
                        # ab_params,
                        # rf_params,
                        # et_params,
                        # gb_params,
                        dnn_params
                        ]

        return layer1_params

    @staticmethod
    def get_layer2_params(train_seed):
        """
            Set Parameters for models of layer2
        """

        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',               # gdbt,rf,dart,goss
                      'learning_rate': 0.003,           # default=0.1
                      'num_leaves': 88,                 # default=31       <2^(max_depth)
                      'max_depth': 10,                   # default=-1
                      'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,            # default=1
                      'feature_fraction_seed': 10,      # default=2
                      'bagging_fraction': 0.8,          # default=1
                      'bagging_freq': 1,                # default=0        perform bagging every k iteration
                      'bagging_seed': 19,               # default=3
                      'lambda_l1': 0,                   # default=0
                      'lambda_l2': 0,                   # default=0
                      'min_gain_to_split': 0,           # default=0
                      'max_bin': 225,                  # default=255
                      'min_data_in_bin': 5,             # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,      # default=0
                      'seed': train_seed}

        # Parameters of Deep Neural Network
        # dnn_params = {'version': '1.0',
        #               'epochs': 10,
        #               'unit_number': [4, 2],
        #               'learning_rate': 0.0001,
        #               'keep_probability': 0.8,
        #               'batch_size': 256,
        #               'seed': train_seed,
        #               'display_step': 100,
        #               'save_path': dnn_save_path,
        #               'log_path': dnn_log_path}

        # List of parameters for layer2
        layer2_params = [
                        lgb_params,
                        # dnn_params
                        ]

        return layer2_params

    @staticmethod
    def get_final_layer_params(train_seed):
        """
            Set Parameters for models of final layer
        """

        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',               # gdbt,rf,dart,goss
                      'learning_rate': 0.003,           # default=0.1
                      'num_leaves': 88,                 # default=31       <2^(max_depth)
                      'max_depth': 10,                   # default=-1
                      'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,            # default=1
                      'feature_fraction_seed': 10,      # default=2
                      'bagging_fraction': 0.8,          # default=1
                      'bagging_freq': 1,                # default=0        perform bagging every k iteration
                      'bagging_seed': 19,               # default=3
                      'lambda_l1': 0,                   # default=0
                      'lambda_l2': 0,                   # default=0
                      'min_gain_to_split': 0,           # default=0
                      'max_bin': 225,                  # default=255
                      'min_data_in_bin': 5,             # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,      # default=0
                      'seed': train_seed}

        # Parameters of Deep Neural Network
        # dnn_params = {'version': '1.0',
        #               'epochs': 10,
        #               'unit_number': [4, 2],
        #               'learning_rate': 0.0001,
        #               'keep_probability': 0.8,
        #               'batch_size': 256,
        #               'seed': train_seed,
        #               'display_step': 100,
        #               'save_path': dnn_save_path,
        #               'log_path': dnn_log_path}

        return lgb_params

    @staticmethod
    def deep_stack_train(train_seed, cv_seed):
        """
            Training model using DeepStack model
        """

        hyper_params = {'n_valid': (4, 4),
                        'n_era': (20, 20),
                        'n_epoch': (3, 1),
                        'train_seed': train_seed,
                        'cv_seed': cv_seed,
                        'num_boost_round_lgb_l1': 65,
                        'num_boost_round_xgb_l1': 50,
                        'num_boost_round_lgb_l2': 65,
                        'num_boost_round_final': 65,
                        'show_importance': False,
                        'show_accuracy': True,
                        'save_epoch_results': False}

        layer1_prams = ModelStacking.get_layer1_params(train_seed)
        layer2_prams = ModelStacking.get_layer2_params(train_seed)
        # layer3_prams = ModelStacking.get_layer3_params(train_seed)

        layers_params = [layer1_prams,
                         layer2_prams,
                         # layer3_prams
                         ]

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        STK = stacking.DeepStack(x_train, y_train, w_train, e_train, x_test, id_test, x_g_train, x_g_test,
                                 pred_path=stack_pred_path, loss_log_path=loss_log_path,
                                 stack_output_path=stack_output_path, hyper_params=hyper_params,
                                 layers_params=layers_params)

        STK.stack()

    @staticmethod
    def stack_tree_train(train_seed, cv_seed, save_auto_train_results=False, idx=None):
        """
            Training model using StackTree model
        """

        if save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        hyper_params = {'n_valid': (4, 4),
                        'n_era': (20, 20),
                        'n_epoch': (1, 8),
                        'final_n_cv': 20,
                        'train_seed': train_seed,
                        'cv_seed': cv_seed,
                        'num_boost_round_lgb_l1': 65,
                        'num_boost_round_xgb_l1': 65,
                        'num_boost_round_final': 65,
                        'show_importance': False,
                        'show_accuracy': True,
                        'save_epoch_results': False}

        layer1_params = ModelStacking.get_layer1_params(train_seed)
        # layer2_params = ModelStacking.get_layer2_params(train_seed)
        final_layer_params = ModelStacking.get_final_layer_params(train_seed)

        layers_params = [layer1_params,
                         # layer2_params,
                         final_layer_params]

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        STK = stacking.StackTree(x_train, y_train, w_train, e_train, x_test, id_test, x_g_train, x_g_test,
                                 layers_params=layers_params, hyper_params=hyper_params)

        STK.stack(pred_path=stack_pred_path, loss_log_path=loss_log_path, stack_output_path=stack_output_path,
                  csv_log_path=csv_log_path+'stack_final_', save_csv_log=True, csv_idx=idx)


def auto_grid_search():
    """
        Automatically Grid Searching
    """

    parameter_grid = ['num_leaves', (80, 85, 90)]
    n_epoch = 100

    for param in parameter_grid[1]:

        param_start_time = time.time()

        grid_search_tuple = (parameter_grid[0], param)

        for i in range(n_epoch):

            train_seed = random.randint(0, 300)
            cv_seed = random.randint(0, 300)
            epoch_start_time = time.time()

            print('======================================================')
            print('Auto Training... | Parameter:{} | Epoch: {}/{}'.format(param, i+1, n_epoch))
            print('======================================================')

            # Logistic Regression
            # TrainSingleModel.lr_train(train_seed, cv_seed, save_auto_train_results=True,
            #                           idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # Random Forest
            # TrainSingleModel.rf_train(train_seed, cv_seed, save_auto_train_results=True,
            #                           idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # Extra Trees
            # TrainSingleModel.et_train(train_seed, cv_seed, save_auto_train_results=True,
            #                           idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # AdaBoost
            # TrainSingleModel.ab_train(train_seed, cv_seed, save_auto_train_results=True,
            #                           idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # GradientBoosting
            # TrainSingleModel.gb_train(train_seed, cv_seed, save_auto_train_results=True,
            #                           idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # XGBoost
            # TrainSingleModel.xgb_train(train_seed, cv_seed, save_auto_train_results=True,
            #                            idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)
            # TrainSingleModel.xgb_train_sklearn(train_seed, cv_seed, save_auto_train_results=True,
            #                                    idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # LightGBM
            TrainSingleModel.lgb_train(train_seed, cv_seed, save_auto_train_results=True,
                                       idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)
            # TrainSingleModel.lgb_train_sklearn(train_seed, cv_seed, save_auto_train_results=True,
            #                                    idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # CatBoost
            # TrainSingleModel.cb_train(train_seed, cv_seed, save_auto_train_results=True,
            #                           idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # DNN
            # TrainSingleModel.dnn_tf_train(train_seed, cv_seed, save_auto_train_results=True,
            #                               idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)
            # TrainSingleModel.dnn_keras_train(train_seed, cv_seed, save_auto_train_results=True,
            #                                  idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            # Champion Model
            # ChampionModel.Christ1991(train_seed, cv_seed, save_auto_train_results=True,
            #                          idx=str(param) + str(i+1), grid_search_tuple=grid_search_tuple)

            print('======================================================')
            print('Auto Training Epoch Done!')
            print('Train Seed: {}'.format(train_seed))
            print('Cross Validation Seed: {}'.format(cv_seed))
            print('Epoch Time: {}s'.format(time.time() - epoch_start_time))
            print('======================================================')

        print('======================================================')
        print('One Parameter Done!')
        print('Epoch Time: {}s'.format(time.time() - param_start_time))
        print('======================================================')


def auto_train():
    """
        Automatically training a model for many times
    """

    n_epoch = 100

    for i in range(n_epoch):

        train_seed = random.randint(0, 300)
        cv_seed = random.randint(0, 300)
        epoch_start_time = time.time()

        print('======================================================')
        print('Auto Training Epoch {}/{}...'.format(i+1, n_epoch))
        print('======================================================')

        # Logistic Regression
        # TrainSingleModel.lr_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)

        # Random Forest
        # TrainSingleModel.rf_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)

        # Extra Trees
        # TrainSingleModel.et_train(train_seed, cv_seed, save_auto_train_results=True, idx=+1)

        # AdaBoost
        # TrainSingleModel.ab_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)

        # GradientBoosting
        # TrainSingleModel.gb_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)

        # XGBoost
        # TrainSingleModel.xgb_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)
        # TrainSingleModel.xgb_train_sklearn(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)

        # LightGBM
        # TrainSingleModel.lgb_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)
        # TrainSingleModel.lgb_train_sklearn(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)

        # CatBoost
        TrainSingleModel.cb_train(train_seed, cv_seed, idx=i+1)

        # DNN
        # TrainSingleModel.dnn_tf_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)
        # TrainSingleModel.dnn_keras_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)

        # Champion Model
        # ChampionModel.Christ1991(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)

        # Stacking
        #  ModelStacking.stack_tree_train(train_seed, cv_seed, save_auto_train_results=True, idx=i+1)
        #  for ii in range(5):
        #      t_seed = random.randint(0, 300)
        #      c_seed = random.randint(0, 300)
        #      TrainSingleModel.stack_lgb_train(t_seed, c_seed, idx='auto_{}_epoch_{}'.format(i+1, ii+1), i+1)

        print('======================================================')
        print('Auto Training Epoch Done!')
        print('Train Seed: {}'.format(train_seed))
        print('Cross Validation Seed: {}'.format(cv_seed))
        print('Epoch Time: {}s'.format(time.time() - epoch_start_time))
        print('======================================================')


if __name__ == "__main__":

    start_time = time.time()

    # Check if directories exit or not
    utils.check_dir(path_list)

    # Create Global Seed for Training and Cross Validation
    global_train_seed = random.randint(0, 300)
    global_cv_seed = random.randint(0, 300)
    # global_train_seed = 65
    # global_cv_seed = 6

    print('======================================================')
    print('Start Training...')

    # Logistic Regression
    # TrainSingleModel.lr_train(global_train_seed, global_cv_seed)

    # Random Forest
    # TrainSingleModel.rf_train(global_train_seed, global_cv_seed)

    # Extra Trees
    # TrainSingleModel.et_train(global_train_seed, global_cv_seed)

    # AdaBoost
    # TrainSingleModel.ab_train(global_train_seed, global_cv_seed)

    # GradientBoosting
    # TrainSingleModel.gb_train(global_train_seed, global_cv_seed)

    # XGBoost
    # TrainSingleModel.xgb_train(global_train_seed, global_cv_seed)
    # TrainSingleModel.xgb_train_sklearn(global_train_seed, global_cv_seed)

    # LightGBM
    # TrainSingleModel.lgb_train(global_train_seed, global_cv_seed)
    # TrainSingleModel.lgb_train_sklearn(global_train_seed, global_cv_seed)

    # CatBoost
    # TrainSingleModel.cb_train(global_train_seed, global_cv_seed)

    # DNN
    # TrainSingleModel.dnn_tf_train(global_train_seed, global_cv_seed)
    # TrainSingleModel.dnn_keras_train(global_train_seed, global_cv_seed)

    # Champion Model
    # ChampionModel.Christ1991(global_train_seed, global_cv_seed)

    # Grid Search
    # GridSearch.lr_grid_search(global_train_seed, global_cv_seed)
    # GridSearch.rf_grid_search(global_train_seed, global_cv_seed)
    # GridSearch.et_grid_search(global_train_seed, global_cv_seed)
    # GridSearch.ab_grid_search(global_train_seed, global_cv_seed)
    # GridSearch.gb_grid_search(global_train_seed, global_cv_seed)
    # GridSearch.xgb_grid_search(global_train_seed, global_cv_seed)
    # GridSearch.lgb_grid_search(global_train_seed, global_cv_seed)
    # GridSearch.stack_lgb_grid_search(global_train_seed, global_cv_seed)

    # Stacking
    # ModelStacking.deep_stack_train(global_train_seed, global_cv_seed)
    # ModelStacking.stack_tree_train(global_train_seed, global_cv_seed)
    # TrainSingleModel.stack_lgb_train(213, 33, auto_idx='2')

    # Prejudge
    PrejudgeTraining.binary_train(global_train_seed, global_cv_seed)
    # PrejudgeTraining.multiclass_train(global_train_seed, global_cv_seed)

    # Auto Training
    # auto_train()

    # Auto Training
    # auto_grid_search()

    print('======================================================')
    print('All Task Done!')
    print('Global Train Seed: {}'.format(global_train_seed))
    print('Global Cross Validation Seed: {}'.format(global_cv_seed))
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')
