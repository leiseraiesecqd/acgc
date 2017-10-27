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


class SingleModel:
    """
        Train single model
    """
    def __init__(self, reduced_feature_list=None, save_auto_train_results=True, grid_search_n_cv=None, options=None):

        self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test\
            = utils.load_preprocessed_data(preprocessed_data_path)
        self.x_g_train, self.x_g_test\
            = utils.load_preprocessed_data_g(preprocessed_data_path)

        # Choose Useful features
        if reduced_feature_list is not None:

            useful_feature_list_g = reduced_feature_list + [-1]
            reduced_feature_list.extend(list(range(-1, -29, -1)))
            self.x_train = self.x_train[:, reduced_feature_list]
            self.x_g_train = self.x_g_train[:, useful_feature_list_g]
            self.x_test = self.x_test[:, reduced_feature_list]
            self.x_g_test = self.x_g_test[:, useful_feature_list_g]

        self.single_model_pred_path = single_model_pred_path
        self.loss_log_path = loss_log_path
        self.csv_log_path = csv_log_path + 'single_'
        self.train_seed = 0
        self.cv_seed = 0
        self.save_auto_train_results = save_auto_train_results
        self.grid_search_n_cv = grid_search_n_cv
        self.options = options

    def train_model(self, model=None, parameters=None, idx=None, cv_generator=None, grid_search_tuple=None):

        if self.save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        if self.grid_search_n_cv is not None:
            n_cv = self.grid_search_n_cv
        else:
            n_cv = 20

        # Parameters for Train
        model.train(pred_path=self.single_model_pred_path, loss_log_path=self.loss_log_path,
                    csv_log_path=self.csv_log_path, n_valid=4, n_cv=n_cv, n_era=20, train_seed=self.train_seed,
                    cv_seed=self.cv_seed, era_list=None, parameters=parameters, csv_idx=idx,
                    cv_generator=cv_generator, auto_train_pred_path=auto_train_path, **self.options)

    def lr_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            Logistic Regression
        """

        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.LRegression(self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def rf_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            Random Forest
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.RandomForest(self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def et_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            Extra Trees
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.ExtraTrees(self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def ab_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            AdaBoost
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.AdaBoost(self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def gb_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            GradientBoosting
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.GradientBoosting(self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def xgb_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None, grid_boost_round=None):
        """
            XGBoost
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

        parameters = {'eta': 0.003,
                      'gamma': 0,                       # 如果loss function小于设定值，停止产生子节点
                      'max_depth': 9,                   # default=6
                      'min_child_weight': 18,           # default=1，建立每个模型所需最小样本权重和
                      'subsample': 0.9,                 # 建立树模型时抽取子样本占整个样本的比例
                      'colsample_bytree': 0.7,          # 建立树时对特征随机采样的比例
                      'colsample_bylevel': 0.6,
                      'lambda': 0,
                      'alpha': 0,
                      'early_stopping_rounds': 30,
                      'n_jobs': -1,
                      'objective': 'binary:logistic',
                      'eval_metric': 'logloss',
                      'seed': train_seed}

        if grid_boost_round is not None:
            num_boost_round = grid_boost_round
        else:
            num_boost_round = 30

        model = models.XGBoost(self.x_train, self.y_train, self.w_train, self.e_train,
                               self.x_test, self.id_test, num_boost_round=num_boost_round)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def xgb_train_sklearn(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            XGBoost using scikit-learn module
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.SKLearnXGBoost(self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def lgb_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None, grid_boost_round=None):
        """
            LightGBM
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

        parameters = {'application': 'binary',
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 83,                     # default=31       <2^(max_depth)
                      'max_depth': 9,                       # default=-1
                      'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 9,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,          # default=0
                      'seed': train_seed}

        if grid_boost_round is not None:
            num_boost_round = grid_boost_round
        else:
            num_boost_round = 108

        model = models.LightGBM(self.x_g_train, self.y_train, self.w_train, self.e_train,
                                self.x_g_test, self.id_test, num_boost_round=num_boost_round)

        # cv_generator = CrossValidation.era_k_fold_with_weight_all_random
        # cv_generator = CrossValidation.random_split_with_weight
        # cv_generator = CrossValidation.era_k_fold_with_weight_balance

        self.train_model(model=model, parameters=parameters, idx=idx,
                         cv_generator=None, grid_search_tuple=grid_search_tuple)

    def lgb_train_sklearn(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            LightGBM using scikit-learn module
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.SKLearnLightGBM(self.x_g_train, self.y_train, self.w_train,
                                       self.e_train, self.x_g_test, self.id_test)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def cb_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            CatBoost
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.CatBoost(self.x_g_train, self.y_train, self.w_train, self.e_train, self.x_g_test, self.id_test)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def dnn_tf_train(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            Deep Neural Networks
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
                                          self.e_train, self.x_test, self.id_test, parameters)

        self.train_model(model=model, parameters=parameters, idx=idx, grid_search_tuple=grid_search_tuple)

    def stack_lgb_train(self, train_seed, cv_seed, idx=None, auto_idx=None, grid_search_tuple=None):
        """
            LightGBM for stack layer
        """
        blender_x_tree, blender_test_tree, blender_x_g_tree, blender_test_g_tree\
            = utils.load_stacked_data(stack_output_path + 'auto_stack_{}_l2_'.format(auto_idx))

        if self.save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        g_train = self.x_g_train[:, -1]
        g_test = self.x_g_test[:, -1]

        x_train_reuse = self.x_train[:, :87]
        x_test_reuse = self.x_test[:, :87]

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
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 83,                     # default=31       <2^(max_depth)
                      'max_depth': 9,                       # default=-1
                      'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 9,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,          # default=0
                      'seed': train_seed}

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]
        if self.grid_search_n_cv is None:
            n_cv = 20
        else:
            n_cv = self.grid_search_n_cv

        LGB = models.LightGBM(blender_x_g_tree, self.y_train, self.w_train, self.e_train,
                              blender_test_g_tree, self.id_test, num_boost_round=72)

        # cv_generator = CrossValidation.era_k_fold_with_weight_balance

        LGB.train(single_model_pred_path, loss_log_path, csv_log_path=csv_log_path + 'stack_final_',
                  n_valid=4, n_cv=n_cv, n_era=20, train_seed=train_seed, cv_seed=cv_seed, parameters=parameters,
                  csv_idx=idx, auto_train_pred_path=auto_train_path, **self.options)


class ChampionModel:

    def __init__(self, reduced_feature_list=None, save_auto_train_results=True, grid_search_n_cv=None, options=None):

        self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test\
            = utils.load_preprocessed_data(preprocessed_data_path)
        self.x_g_train, self.x_g_test\
            = utils.load_preprocessed_data_g(preprocessed_data_path)

        # Choose Useful features
        if reduced_feature_list is not None:
            useful_feature_list_g = reduced_feature_list + [-1]
            reduced_feature_list.extend(list(range(-1, -29, -1)))
            self.x_train = self.x_train[:, reduced_feature_list]
            self.x_g_train = self.x_g_train[:, useful_feature_list_g]
            self.x_test = self.x_test[:, reduced_feature_list]
            self.x_g_test = self.x_g_test[:, useful_feature_list_g]

        self.single_model_pred_path = single_model_pred_path
        self.loss_log_path = loss_log_path
        self.csv_log_path = csv_log_path + 'christ1991_'
        self.train_seed = 0
        self.cv_seed = 0
        self.save_auto_train_results = save_auto_train_results
        self.grid_search_n_cv = grid_search_n_cv
        self.options = options

    def train_model(self, model=None, parameters=None, idx=None, cv_generator=None, grid_search_tuple=None):

        if self.save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        # Grid Search
        if grid_search_tuple is not None:
            parameters[grid_search_tuple[0]] = grid_search_tuple[1]

        if self.grid_search_n_cv is not None:
            n_cv = self.grid_search_n_cv
        else:
            n_cv = 20

        # Parameters for Train
        model.train(pred_path=self.single_model_pred_path, loss_log_path=self.loss_log_path,
                    csv_log_path=self.csv_log_path, n_valid=4, n_cv=n_cv, n_era=20, train_seed=self.train_seed,
                    cv_seed=self.cv_seed, era_list=None, parameters=parameters, csv_idx=idx,
                    cv_generator=cv_generator, auto_train_pred_path=auto_train_path, **self.options)

    def Christ1991(self, train_seed, cv_seed, idx=None, grid_search_tuple=None):
        """
            Model of week3 champion
        """
        self.train_seed = train_seed
        self.cv_seed = cv_seed

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

        model = models.LightGBM(self.x_g_train, self.y_train, self.w_train, self.e_train,
                                self.x_g_test, self.id_test, num_boost_round=65)

        cv_generator = CrossValidation.era_k_fold_with_weight_all_random

        self.train_model(model=model, parameters=parameters, idx=idx,
                         cv_generator=cv_generator, grid_search_tuple=grid_search_tuple)


class SKGridSearch:
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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(
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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)

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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_data_g(preprocessed_data_path)

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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_data_g(preprocessed_data_path)
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
    def __init__(self, load_pickle=False, options=None):
        
        self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test\
            = utils.load_preprocessed_data(preprocessed_data_path)
        self.x_g_train, self.x_g_test\
            = utils.load_preprocessed_data_g(preprocessed_data_path)
        self.x_train_p, self.y_train_p, self.w_train_p, self.e_train_p, self.x_g_train_p \
            = utils.load_preprocessed_positive_data(preprocessed_data_path)
        self.x_train_n, self.y_train_n, self.w_train_n, self.e_train_n, self.x_g_train_n \
            = utils.load_preprocessed_negative_data(preprocessed_data_path)
        self.load_pickle = load_pickle
        self.show_importance = options['show_importance']
        self.show_accuracy = options['show_accuracy']

    @staticmethod
    def get_binary_models_parameters(train_seed):
        """
            Set Parameters for models of PrejudgeBinary
        """
        era_training_params = {'application': 'binary',
                               'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                               'learning_rate': 0.1,                 # default=0.1
                               'num_leaves': 75,                     # default=31       <2^(max_depth)
                               'max_depth': 9,                       # default=-1
                               'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                               'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                               'feature_fraction': 0.5,              # default=1
                               'feature_fraction_seed': train_seed,  # default=2
                               'bagging_fraction': 0.8,              # default=1
                               'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                               'bagging_seed': train_seed,           # default=3
                               'lambda_l1': 0,                       # default=0
                               'lambda_l2': 0,                       # default=0
                               'min_gain_to_split': 0,               # default=0
                               'max_bin': 225,                       # default=255
                               'min_data_in_bin': 9,                 # default=5
                               'metric': 'binary_logloss',
                               'num_threads': -1,
                               'verbosity': 1,
                               'early_stopping_rounds': 50,          # default=0
                               'seed': train_seed}

        positive_params = {'application': 'binary',
                           'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                           'learning_rate': 0.003,               # default=0.1
                           'num_leaves': 75,                     # default=31       <2^(max_depth)
                           'max_depth': 9,                       # default=-1
                           'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                           'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                           'feature_fraction': 0.5,              # default=1
                           'feature_fraction_seed': train_seed,  # default=2
                           'bagging_fraction': 0.8,              # default=1
                           'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                           'bagging_seed': train_seed,           # default=3
                           'lambda_l1': 0,                       # default=0
                           'lambda_l2': 0,                       # default=0
                           'min_gain_to_split': 0,               # default=0
                           'max_bin': 225,                       # default=255
                           'min_data_in_bin': 9,                 # default=5
                           'metric': 'binary_logloss',
                           'num_threads': -1,
                           'verbosity': 1,
                           'early_stopping_rounds': 50,          # default=0
                           'seed': train_seed}

        negative_params = {'application': 'binary',
                           'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                           'learning_rate': 0.003,               # default=0.1
                           'num_leaves': 75,                     # default=31       <2^(max_depth)
                           'max_depth': 9,                       # default=-1
                           'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                           'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                           'feature_fraction': 0.5,              # default=1
                           'feature_fraction_seed': train_seed,  # default=2
                           'bagging_fraction': 0.8,              # default=1
                           'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                           'bagging_seed': train_seed,           # default=3
                           'lambda_l1': 0,                       # default=0
                           'lambda_l2': 0,                       # default=0
                           'min_gain_to_split': 0,               # default=0
                           'max_bin': 225,                       # default=255
                           'min_data_in_bin': 9,                 # default=5
                           'metric': 'binary_logloss',
                           'num_threads': -1,
                           'verbosity': 1,
                           'early_stopping_rounds': 50,          # default=0
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

    def binary_train(self, train_seed, cv_seed):
        """
            Training model of PrejudgeBinary
        """
        
        models_parameters = PrejudgeTraining.get_binary_models_parameters(train_seed)

        positive_era_list = preprocess.positive_era_list
        negative_era_list = preprocess.negative_era_list

        hyper_parameters = {'cv_seed': cv_seed,
                            'train_seed': train_seed,
                            'n_splits_e': 10,
                            'num_boost_round_e': 3000,
                            'n_cv_e': 10,
                            'n_valid_p': 2,
                            'n_cv_p': 12,
                            'n_era_p': len(positive_era_list),
                            'num_boost_round_p': 80,
                            'era_list_p': positive_era_list,
                            'n_valid_n': 1,
                            'n_cv_n': 8,
                            'n_era_n': len(negative_era_list),
                            'num_boost_round_n': 80,
                            'era_list_n': negative_era_list,
                            'force_convert_era': True,
                            'use_weight': True,
                            'show_importance': self.show_importance,
                            'show_accuracy': self.show_accuracy}

        PB = prejudge.PrejudgeBinary(self.x_train, self.y_train, self.w_train, self.e_train, self.x_g_train, 
                                     self.x_train_p, self.y_train_p, self.w_train_p, self.e_train_p, self.x_g_train_p, 
                                     self.x_train_n, self.y_train_n, self.w_train_n, self.e_train_n, self.x_g_train_n, 
                                     self.x_test, self.id_test, self.x_g_test,
                                     pred_path=prejudge_pred_path, prejudged_data_path=prejudged_data_path,
                                     loss_log_path=prejudge_loss_log_path, csv_log_path=csv_log_path + 'prejudge_',
                                     models_parameters=models_parameters, hyper_parameters=hyper_parameters)

        PB.train(load_pickle=self.load_pickle, load_pickle_path=None)

    def multiclass_train(self, train_seed, cv_seed):
        """
            Training model of PrejudgeMultiClass
        """

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
                            'show_importance': self.show_importance,
                            'show_accuracy': self.show_accuracy}

        PB = prejudge.PrejudgeMultiClass(self.x_train, self.y_train, self.w_train, self.e_train, self.x_g_train,
                                         self.x_test, self.id_test, self.x_g_test,
                                         pred_path=prejudge_pred_path, prejudged_data_path=prejudged_data_path,
                                         loss_log_path=prejudge_loss_log_path, csv_log_path=csv_log_path + 'prejudge_',
                                         models_parameters=models_parameters, hyper_parameters=hyper_parameters)

        PB.train(load_pickle=self.load_pickle, load_pickle_path=None)


class ModelStacking:
    """
        Stacking
    """
    def __init__(self, reduced_feature_list=None, save_auto_train_results=True, options=None):

        self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test\
            = utils.load_preprocessed_data(preprocessed_data_path)
        self.x_g_train, self.x_g_test\
            = utils.load_preprocessed_data_g(preprocessed_data_path)

        # Choose Useful features
        if reduced_feature_list is not None:
            useful_feature_list_g = reduced_feature_list + [-1]
            reduced_feature_list.extend(list(range(-1, -29, -1)))
            self.x_train = self.x_train[:, reduced_feature_list]
            self.x_g_train = self.x_g_train[:, useful_feature_list_g]
            self.x_test = self.x_test[:, reduced_feature_list]
            self.x_g_test = self.x_g_test[:, useful_feature_list_g]

        self.save_auto_train_results = save_auto_train_results
        self.options = options

    @staticmethod
    def get_layer1_params(train_seed):
        """
            Set Parameters for models of layer1
        """
        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 83,                     # default=31       <2^(max_depth)
                      'max_depth': 9,                       # default=-1
                      'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 9,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,          # default=0
                      'seed': train_seed}

        # Parameters of XGBoost
        xgb_params = {'eta': 0.003,
                      'gamma': 0,                       # 如果loss function小于设定值，停止产生子节点
                      'max_depth': 9,                   # default=6
                      'min_child_weight': 18,           # default=1，建立每个模型所需最小样本权重和
                      'subsample': 0.9,                 # 建立树模型时抽取子样本占整个样本的比例
                      'colsample_bytree': 0.8,          # 建立树时对特征随机采样的比例
                      'colsample_bylevel': 0.7,
                      'lambda': 0,
                      'alpha': 0,
                      'early_stopping_rounds': 30,
                      'n_jobs': -1,
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
                      'epochs': 4,
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
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 83,                     # default=31       <2^(max_depth)
                      'max_depth': 9,                       # default=-1
                      'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 9,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,          # default=0
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
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 83,                     # default=31       <2^(max_depth)
                      'max_depth': 9,                       # default=-1
                      'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 9,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,          # default=0
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
                        'num_boost_round_xgb_l1': 30,
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

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_data_g(preprocessed_data_path)

        STK = stacking.DeepStack(x_train, y_train, w_train, e_train, x_test, id_test, x_g_train, x_g_test,
                                 pred_path=stack_pred_path, loss_log_path=loss_log_path,
                                 stack_output_path=stack_output_path, hyper_params=hyper_params,
                                 layers_params=layers_params)

        STK.stack()

    def stack_tree_train(self, train_seed, cv_seed, idx=None):
        """
            Training model using StackTree model
        """
        if self.save_auto_train_results is True:
            auto_train_path = auto_train_pred_path
        else:
            auto_train_path = None

        hyper_params = {'n_valid': (4, 4),
                        'n_era': (20, 20),
                        'n_epoch': (1, 8),
                        'final_n_cv': 20,
                        'train_seed': train_seed,
                        'cv_seed': cv_seed,
                        'models_l1': ('lgb', 'xgb', 'dnn'),
                        'models_l2': (),
                        'model_final': 'lgb',
                        'num_boost_round_lgb_l1': 108,
                        'num_boost_round_xgb_l1': 115,
                        'num_boost_round_final': 80,
                        'useful_feature_list_l1': None,
                        'reuse_feature_list_final': range(87),
                        'scale_blender_final': True,
                        'save_epoch_results': False}

        layer1_params = ModelStacking.get_layer1_params(train_seed)
        # layer2_params = ModelStacking.get_layer2_params(train_seed)
        final_layer_params = ModelStacking.get_final_layer_params(train_seed)

        layers_params = [layer1_params,
                         # layer2_params,
                         final_layer_params]

        STK = stacking.StackTree(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.id_test, self.x_g_train, self.x_g_test,
                                 layers_params=layers_params, hyper_params=hyper_params, options=self.options)

        STK.stack(pred_path=stack_pred_path, auto_train_pred_path=auto_train_path, loss_log_path=loss_log_path,
                  stack_output_path=stack_output_path, csv_log_path=csv_log_path+'stack_final_', csv_idx=idx)
        

class Training:
    
    def __init__(self):
        pass

    @staticmethod
    def get_train_function(train_mode, model_name, grid_search_n_cv=None,
                           reduced_feature_list=None, load_pickle=False, options=None):

        if train_mode == 'train_single_model':
            model_arg = {'reduced_feature_list': reduced_feature_list,
                         'save_auto_train_results': False, 'options': options}
        elif train_mode == 'auto_grid_search':
            options['save_final_pred'] = False
            model_arg = {'reduced_feature_list': reduced_feature_list, 'save_auto_train_results': False,
                         'grid_search_n_cv': grid_search_n_cv, 'options': options}
        elif train_mode == 'auto_grid_boost_round':
            model_arg = {'reduced_feature_list': reduced_feature_list, 'save_auto_train_results': True,
                         'grid_search_n_cv': grid_search_n_cv, 'options': options}
        elif train_mode == 'auto_train':
            model_arg = {'reduced_feature_list': reduced_feature_list,
                         'save_auto_train_results': True, 'options': options}
        else:
            raise ValueError('Wrong Training Mode!')

        if model_name in ['lr', 'rf', 'et', 'ab', 'gb', 'xgb', 'xgb_sk',
                          'lgb', 'lgb_sk', 'cb', 'dnn', 'stack_lgb']:

            SM = SingleModel(**model_arg)
            train_functions = {'lr': SM.lr_train,
                               'rf': SM.rf_train,
                               'et': SM.et_train,
                               'ab': SM.ab_train,
                               'gb': SM.gb_train,
                               'xgb': SM.xgb_train,
                               'xgb_sk': SM.xgb_train_sklearn,
                               'lgb': SM.lgb_train,
                               'lgb_sk': SM.lgb_train_sklearn,
                               'cb': SM.cb_train,
                               'dnn': SM.dnn_tf_train,
                               'stack_lgb': SM.stack_lgb_train}

            return train_functions[model_name]

        elif model_name == 'christ':
            CM = ChampionModel(**model_arg)
            return CM.Christ1991

        elif model_name == 'stack_t':
            STK = ModelStacking(**model_arg)
            return STK.stack_tree_train

        elif model_name == 'prejudge_b':
            PJ = PrejudgeTraining(load_pickle=load_pickle, options=options)
            return PJ.binary_train

        elif model_name == 'prejudge_m':
            PJ = PrejudgeTraining(load_pickle=load_pickle, options=options)
            return PJ.multiclass_train

        else:
            raise ValueError('Wrong Model Name!')

    def train_single_model(self, model_name, train_seed, cv_seed, auto_idx=None,
                           reduced_feature_list=None, load_pickle=False, options=None):
        """
            Training Single Model
        """
        # Get Train Function
        train_function = self.get_train_function('train_single_model', model_name, load_pickle=load_pickle,
                                                 reduced_feature_list=reduced_feature_list, options=options)

        # Training Model
        if model_name == 'stack_lgb':
            train_function(train_seed, cv_seed, auto_idx=auto_idx)
        else:
            train_function(train_seed, cv_seed)

    def auto_grid_search(self, model_name=None, parameter_grid_list=None,
                         reduced_feature_list=None, n_epoch=1, grid_search_n_cv=5, options=None):
        """
            Automatically Grid Searching
        """
        # Get Train Function
        train_function = self.get_train_function('auto_grid_search', model_name, grid_search_n_cv=grid_search_n_cv,
                                                 reduced_feature_list=reduced_feature_list, options=options)
    
        for parameter_grid in parameter_grid_list:
    
            gs_start_time = time.time()
    
            print('======================================================')
            print('Auto Grid Searching Parameter: {}'.format(parameter_grid[0]))
            print('======================================================')
    
            for param in parameter_grid[1]:
    
                param_start_time = time.time()
                grid_search_tuple = (parameter_grid[0], param)
    
                for i in range(n_epoch):
    
                    train_seed = random.randint(0, 300)
                    cv_seed = random.randint(0, 300)
                    epoch_start_time = time.time()
                    idx = parameter_grid[0] + '_' + str(param) + '_' + str(i+1)
    
                    print('======================================================')
                    print('Parameter: {}-{} | Epoch: {}/{} | train_seed: {} | cv_seed: {}'
                          .format(parameter_grid[0], param, i+1, n_epoch, train_seed, cv_seed))

                    # Training Model
                    train_function(train_seed, cv_seed, idx=idx, grid_search_tuple=grid_search_tuple)
    
                    print('======================================================')
                    print('Auto Training Epoch Done!')
                    print('Train Seed: {}'.format(train_seed))
                    print('Cross Validation Seed: {}'.format(cv_seed))
                    print('Epoch Time: {}s'.format(time.time() - epoch_start_time))
                    print('======================================================')
    
                print('======================================================')
                print('One Parameter Done!')
                print('Parameter Time: {}s'.format(time.time() - param_start_time))
                print('======================================================')
    
            print('======================================================')
            print('All Parameter Done!')
            print('Grid Searching Time: {}s'.format(time.time() - gs_start_time))
            print('======================================================')

    def auto_grid_boost_round(self, model_name=None, grid_boost_round_tuple=None, reduced_feature_list=None,
                              train_seed_list=None, cv_seed_list=None, n_epoch=1, grid_search_n_cv=20, options=None):
        """
            Automatically Grid Searching
        """

        def _random_int_list(start, stop, length):
            start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
            length = int(abs(length)) if length else 0
            random_list = []
            for _ in range(length):
                random_list.append(random.randint(start, stop))
            return random_list

        if train_seed_list is None:
            train_seed_list = _random_int_list(0, 500, n_epoch)
            cv_seed_list = _random_int_list(0, 500, n_epoch)

        # Get Train Function
        train_function = self.get_train_function('auto_grid_boost_round', model_name, grid_search_n_cv=grid_search_n_cv,
                                                 reduced_feature_list=reduced_feature_list, options=options)

        print('======================================================')
        print('Auto Grid Searching num_boost_round...')
        print('======================================================')

        for num_boost_round in grid_boost_round_tuple:

            param_start_time = time.time()

            for i, (train_seed, cv_seed) in enumerate(zip(train_seed_list, cv_seed_list)):

                epoch_start_time = time.time()
                idx = 'boost_round_' + str(num_boost_round) + '_' + str(i + 1)

                print('======================================================')
                print('num_boost_round: {} | Epoch: {}/{} | train_seed: {} | cv_seed: {}'
                      .format(num_boost_round, i + 1, n_epoch, train_seed, cv_seed))

                # Training Model
                train_function(train_seed, cv_seed, idx=idx, grid_boost_round=num_boost_round)

                print('======================================================')
                print('Auto Training Epoch Done!')
                print('Train Seed: {}'.format(train_seed))
                print('Cross Validation Seed: {}'.format(cv_seed))
                print('Epoch Time: {}s'.format(time.time() - epoch_start_time))
                print('======================================================')

            print('======================================================')
            print('One Parameter Done!')
            print('Parameter Time: {}s'.format(time.time() - param_start_time))
            print('======================================================')

    def auto_train(self, model_name=None, reduced_feature_list=None,
                   n_epoch=1, stack_final_epochs=None, options=None):
        """
            Automatically training a model for many times
        """

        # Get Train Function
        train_function = self.get_train_function('auto_train', model_name,
                                                 reduced_feature_list=reduced_feature_list, options=options)
    
        for i in range(n_epoch):
    
            train_seed = random.randint(0, 500)
            cv_seed = random.randint(0, 500)
            epoch_start_time = time.time()
    
            print('======================================================')
            print('Auto Training Epoch {}/{}...'.format(i+1, n_epoch))
    
            if model_name == 'stack_t':
                train_function(train_seed, cv_seed, idx=i+1)
                train_function_s = self.get_train_function('auto_train', 'stack_lgb',
                                                           reduced_feature_list=reduced_feature_list, options=options)
                for ii in range(stack_final_epochs):
                    t_seed = random.randint(0, 500)
                    c_seed = random.randint(0, 500)
                    train_function_s(t_seed, c_seed, idx='auto_{}_epoch_{}'.format(i+1, ii+1), auto_idx=i+1)
            else:
                train_function(train_seed, cv_seed, idx=i+1)
    
            print('======================================================')
            print('Auto Training Epoch Done!')
            print('Train Seed: {}'.format(train_seed))
            print('Cross Validation Seed: {}'.format(cv_seed))
            print('Epoch Time: {}s'.format(time.time() - epoch_start_time))
            print('======================================================')
    
    def train(self):
        """
            Model Name:
            'lr':           Logistic Regression
            'rf':           Random Forest
            'et':           Extra Trees
            'ab':           AdaBoost
            'gb':           GradientBoosting
            'xgb':          XGBoost
            'xgb_sk':       XGBoost using scikit-learn module
            'lgb':          LightGBM
            'lgb_sk':       LightGBM using scikit-learn module
            'cb':           CatBoost
            'dnn':          Deep Neural Networks
            'stack_lgb':    LightGBM for stack layer
            'christ':       Christ1991
            'prejudge_b':   PrejudgeBinary
            'prejudge_m':   PrejudgeMultiClass
            'stack_t':      StackTree
        """

        # Create Global Seed for Training and Cross Validation
        train_seed = random.randint(0, 500)
        cv_seed = random.randint(0, 500)
        # train_seed = 65
        # cv_seed = 6

        # Training Options
        options = {'show_importance': False,
                   'show_accuracy': True,
                   'save_final_pred': True,
                   'save_final_prob_train': False,
                   'save_cv_pred': False,
                   'save_cv_prob_train': False,
                   'save_csv_log': True}

        # Reduced Features
        reduced_feature_list = None

        """
            Train Single Model
        """
        # self.train_single_model('lgb', train_seed, cv_seed,
        #                         reduced_feature_list=reduced_feature_list, options=options)
        # self.train_single_model('prejudge_b', train_seed, cv_seed, load_pickle=False,
        #                         reduced_feature_list=reduced_feature_list, options=options)

        # self.train_single_model('stack_lgb', train_seed, cv_seed, auto_idx=1,
        #                         reduced_feature_list=reduced_feature_list,  options=options)

        """
            Auto Grid Search Number of Boost Round
        """
        # train_seed_list = [493, 218, 496, 106, 395]
        # cv_seed_list = [35, 73, 288, 325, 458]
        # grid_boost_round_tuple = tuple(range(96, 119, 2))
        # self.auto_grid_boost_round('lgb', grid_boost_round_tuple=grid_boost_round_tuple,
        #                            # train_seed_list=train_seed_list, cv_seed_list=cv_seed_list,
        #                            reduced_feature_list=reduced_feature_list,
        #                            n_epoch=5, grid_search_n_cv=20, options=options)

        """
            Auto Grid Search Parameters
        """
        # pg_list = [
        #            # ['min_child_weight', (15, 18, 21, 24)],
        #            # ['feature_fraction', (0.5, 0.6, 0.7, 0.8, 0.9)],
        #            # ['bagging_fraction', (0.6, 0.7, 0.8, 0.9)],
        #            # ['bagging_freq', (1, 2, 3, 4, 5)],
        #            # ['max_depth', (7, 8, 9, 10)],
        #            ['num_leaves', (75, 77, 79, 81, 83, 85)],
        #            # ['min_data_in_bin', (1, 3, 5, 7, 9)]
        #            ]
        # self.auto_grid_search('lgb', parameter_grid_list=pg_list, n_epoch=200,
        #                       reduced_feature_list=reduced_feature_list,
        #                       grid_search_n_cv=5, options=options)

        """
            Auto Train
        """
        self.auto_train('xgb', n_epoch=200, options=options)
        # self.auto_train('stack_t', n_epoch=5, stack_final_epochs=10,
        #                 reduced_feature_list=reduced_feature_list, options=options)

        print('======================================================')
        print('Global Train Seed: {}'.format(train_seed))
        print('Global Cross Validation Seed: {}'.format(cv_seed))


if __name__ == "__main__":

    start_time = time.time()

    # Check if directories exit or not
    utils.check_dir(path_list)

    print('======================================================')
    print('Start Training...')

    T = Training()
    T.train()

    print('------------------------------------------------------')
    print('All Tasks Done!')
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')
