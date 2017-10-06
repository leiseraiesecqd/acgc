import time
import utils
import models
import stacking
import prejudge
import preprocess
import numpy as np


preprocessed_data_path = './preprocessed_data/'
pred_path = './results/'
grid_search_log_path = './grid_search_logs/'
loss_log_path = './loss_logs/'
stack_output_path = './stacking_outputs/'

path_list = [pred_path,
             pred_path + 'stack_results/',
             pred_path + 'stack_results/epochs_results/',
             pred_path + 'stack_outputs/',
             pred_path + 'stack_outputs/final_results/',
             pred_path + 'prejudge/',
             grid_search_log_path,
             loss_log_path,
             loss_log_path + 'prejudge/',
             stack_output_path]

train_seed = np.random.randint(100)
cv_seed = np.random.randint(100)
dnn_seed = np.random.randint(100)
# train_seed = 65
# cv_seed = 6
# dnn_seed = 21


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
                         'random_state': train_seed,
                         'verbose': 1,
                         'warm_start': False}

        LR = models.LRegression(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training Logistic Regression...')

        LR.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=lr_parameters)

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
                         'random_state': train_seed,
                         'verbose': 2,
                         'warm_start': False}

        RF = models.RandomForest(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training Random Forest...')

        RF.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=rf_parameters)

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
                         'random_state': train_seed,
                         'verbose': 2,
                         'warm_start': False}

        ET = models.ExtraTrees(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training Extra Trees...')

        ET.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=et_parameters)

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
                         'random_state': train_seed,
                         'verbose': 2,
                         'warm_start': False}

        clf_et = models.ExtraTreesClassifier(**et_parameters)

        ab_parameters = {'algorithm': 'SAMME.R',
                         'base_estimator': clf_et,
                         'learning_rate': 0.0051,
                         'n_estimators': 9,
                         'random_state': train_seed}

        AB = models.AdaBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training AdaBoost...')

        AB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=ab_parameters)

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
                         'random_state': train_seed,
                         'subsample': 0.8,
                         'verbose': 2,
                         'warm_start': False}

        GB = models.GradientBoosting(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training GradientBoosting...')

        GB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=gb_parameters)

    # XGBoost
    @staticmethod
    def xgb_train():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        xgb_parameters = {'eta': 0.008,
                          'gamma': 0,                       # 如果loss function小于设定值，停止产生子节点
                          'max_depth': 7,                   # default=6
                          'min_child_weight': 15,           # default=1，建立每个模型所需最小样本权重和
                          'subsample': 0.8,                 # 建立树模型时抽取子样本占整个样本的比例
                          'colsample_bytree': 1,            # 建立树时对特征随机采样的比例
                          'colsample_bylevel': 1,
                          'lambda': 5000,
                          'alpha': 0,
                          'early_stopping_rounds': 30,
                          'nthread': -1,
                          'objective': 'binary:logistic',
                          'eval_metric': 'logloss',
                          'seed': train_seed}

        XGB = models.XGBoost(x_train, y_train, w_train, e_train, x_test, id_test, num_boost_round=50)

        print('Start training XGBoost...')

        XGB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=xgb_parameters)

    @staticmethod
    def xgb_train_sklearn():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        #  x_train_p, y_train_p, w_train_p, e_train_p, x_g_train_p \
        #     = utils.load_preprocessed_positive_pd_data(preprocessed_data_path)
        #  x_train_n, y_train_n, w_train_n, e_train_n, x_g_train_n \
        #      = utils.load_preprocessed_negative_pd_data(preprocessed_data_path)

        xgb_parameters = {'max_depth': 3,
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

        XGB = models.SKLearnXGBoost(x_train, y_train, w_train, e_train, x_test, id_test)

        print('Start training XGBoost...')

        XGB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=xgb_parameters)

        # LGBM = models.SKLearnXGBoost(x_train_n, y_train_n, w_train_n, e_train_n, x_test, id_test)
        #
        # print('Start training XGBoost...')
        #
        # LGBM.train(pred_path, loss_log_path, n_valid=1, n_cv=6, n_era=6, cv_seed=cv_seed, parameters=xgb_parameters)

    # LightGBM
    @staticmethod
    def lgb_train():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        lgb_parameters = {'application': 'binary',
                          'boosting': 'gbdt',               # gdbt,rf,dart,goss
                          'learning_rate': 0.003,           # default=0.1
                          'num_leaves': 88,                 # default=31     <2^(max_depth)
                          'max_depth': 7,                   # default=-1
                          'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                          'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3      reduce over-fit
                          'feature_fraction': 1,            # default=1
                          'feature_fraction_seed': 10,      # default=2
                          'bagging_fraction': 0.8,          # default=1
                          'bagging_freq': 1,                # default=0 perform bagging every k iteration
                          'bagging_seed': 19,               # default=3
                          'lambda_l1': 0,                   # default=0
                          'lambda_l2': 0,                   # default=0
                          'min_gain_to_split': 0,           # default=0
                          'max_bin': 2250,                  # default=255
                          'min_data_in_bin': 5,             # default=5
                          'metric': 'binary_logloss',
                          'num_threads': -1,
                          'verbosity': 1,
                          'early_stopping_rounds': 50,      # default=0
                          'seed': train_seed}

        LGBM = models.LightGBM(x_train, y_train, w_train, e_train, x_test,
                               id_test, x_g_train, x_g_test, num_boost_round=65)

        print('Start training LGBM...')

        LGBM.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=lgb_parameters)

    @staticmethod
    def lgb_train_sklearn():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)
        # x_train_p, y_train_p, w_train_p, e_train_p, x_g_train_p \
        #     = utils.load_preprocessed_positive_pd_data(preprocessed_data_path)
        # x_train_n, y_train_n, w_train_n, e_train_n, x_g_train_n \
        #     = utils.load_preprocessed_negative_pd_data(preprocessed_data_path)

        lgb_parameters = {'learning_rate': 0.003,
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

        LGBM = models.SKLearnLightGBM(x_train, y_train, w_train, e_train, x_test, id_test, x_g_train, x_g_test)

        print('Start training LGBM...')

        LGBM.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=lgb_parameters)

        # LGBM = models.SKLearnLightGBM(x_train_n, y_train_n, w_train_n, e_train_n,
        #                               x_test, id_test, x_g_train_n, x_g_test)
        #
        # print('Start training LGBM...')
        #
        # LGBM.train(pred_path, loss_log_path, n_valid=1, n_cv=6, n_era=6,
        #            cv_seed=cv_seed, era_list=[1, 3, 4, 10, 12, 16], parameters=lgb_parameters)

    # DNN
    @staticmethod
    def dnn_tf_train():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)

        # HyperParameters
        hyper_parameters = {'version': '1.0',
                            'epochs': 4,
                            'unit_number': [2048],
                            'learning_rate': 0.00001,
                            'keep_probability': 0.5,
                            'batch_size': 128,
                            'seed': dnn_seed,
                            'display_step': 100,
                            'save_path': './checkpoints/',
                            'log_path': './log/'}

        print('Loading data set...')

        dnn = models.DeepNeuralNetworks(x_train, y_train, w_train, e_train, x_test, id_test, hyper_parameters)

        print('Start training DNN(TensorFlow)...')

        dnn.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed)

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
    #     dnn.train(pred_path, loss_log_path, n_valid=4, n_cv=20, cv_seed=cv_seed)

    @staticmethod
    def stack_lgb_train():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)
        blender_x_tree, blender_test_tree, blender_x_g_tree, blender_test_g_tree\
            = utils.load_stacked_data(stack_output_path + 'l2_')

        g_train = x_g_train[:, -1]
        g_test = x_g_test[:, -1]

        x_train_reuse = x_train[:, :88]
        x_test_reuse = x_test[:, :88]

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

        lgb_parameters = {'application': 'binary',
                          'boosting': 'gbdt',               # gdbt,rf,dart,goss
                          'learning_rate': 0.002,           # default=0.1
                          'num_leaves': 88,                 # default=31     <2^(max_depth)
                          'max_depth': 7,                   # default=-1
                          'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                          'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3      reduce over-fit
                          'feature_fraction': 1,            # default=1
                          'feature_fraction_seed': 10,      # default=2
                          'bagging_fraction': 0.8,          # default=1
                          'bagging_freq': 1,                # default=0 perform bagging every k iteration
                          'bagging_seed': 19,               # default=3
                          'lambda_l1': 0,                   # default=0
                          'lambda_l2': 0,                   # default=0
                          'min_gain_to_split': 0,           # default=0
                          'max_bin': 2250,                  # default=255
                          'min_data_in_bin': 5,             # default=5
                          'metric': 'binary_logloss',
                          'num_threads': -1,
                          'verbosity': 1,
                          'early_stopping_rounds': 50,      # default=0
                          'seed': train_seed}

        LGB = models.LightGBM(blender_x_tree, y_train, w_train, e_train, blender_test_tree, id_test,
                              blender_x_g_tree, blender_test_g_tree, num_boost_round=70)

        print('Start training LGBM...')

        LGB.train(pred_path, loss_log_path, n_valid=4, n_cv=20, n_era=20, cv_seed=cv_seed, parameters=lgb_parameters)


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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Extra Trees', parameters, parameters_grid)

    # AdaBoost
    @staticmethod
    def ab_grid_search():

        log_path = grid_search_log_path + 'ab_'

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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('XGBoost', parameters, parameters_grid)

    # LightGBM
    @staticmethod
    def lgb_grid_search():

        log_path = grid_search_log_path + 'lgb_'

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

        LGB = models.SKLearnLightGBM(x_train, y_train, w_train, e_train, x_test, id_test, x_g_train, x_g_test)

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

        models.grid_search(log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)

    # Stacking Layer LightGBM
    @staticmethod
    def stack_lgb_grid_search():

        log_path = grid_search_log_path + 'stk_lgb_'

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_outputs, test_outputs, x_g_outputs, test_g_outputs = utils.load_stacked_data(stack_output_path + 'l1_')

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

        LGB = models.SKLearnLightGBM(x_outputs, y_train, w_train, e_train,
                                     test_outputs, id_test, x_g_outputs, test_g_outputs)

        clf = LGB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
            'learning_rate': (0.002, 0.005, 0.01),
            'n_estimators': (30, 60, 90),
            'num_leaves': (32, 64, 128),  # <2^(max_depth)
            'colsample_bytree': (0.6, 0.8, 0.1),
            'max_depth': (6, 8, 10),  # default=-1
            # 'min_data_in_leaf': 20,                  # default=20
            # 'bagging_fraction': (0.5, 0.7, 0.9),
            # 'feature_fraction': (0.5, 0.7, 0.9),
            # 'subsample_for_bin': (50000, 100000, 150000),
            # 'subsample_freq': (4, 6, 8),
            # 'subsample': (0.6, 0.8, 1.0),
            # 'max_bin': (255, 355, 455)
        }

        models.grid_search(log_path, x_outputs, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                           cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)


# Training by Split Era sign
class PrejudgeTraining:

    @staticmethod
    def get_models_parameters():

        era_training_params = {'application': 'binary',
                               'learning_rate': 0.05,
                               'num_leaves': 80,               # <2^(max_depth)
                               'tree_learner': 'serial',
                               'max_depth': 7,                 # default=-1
                               'min_data_in_leaf': 2000,         # default=20
                               'feature_fraction': 0.5,        # default=1
                               'bagging_fraction': 0.6,        # default=1
                               'bagging_freq': 5,              # default=0 perform bagging every k iteration
                               'bagging_seed': 1,              # default=3
                               'early_stopping_rounds': 50,
                               'max_bin': 50,
                               'metric': 'binary_logloss',
                               'verbosity': 1,
                               'seed': train_seed}

        positive_params = {'application': 'binary',
                           'learning_rate': 0.003,
                           'num_leaves': 80,               # <2^(max_depth)
                           'tree_learner': 'serial',
                           'max_depth': 7,                 # default=-1
                           'min_data_in_leaf': 2000,         # default=20
                           'feature_fraction': 0.5,        # default=1
                           'bagging_fraction': 0.6,        # default=1
                           'bagging_freq': 5,              # default=0 perform bagging every k iteration
                           'bagging_seed': 1,              # default=3
                           'early_stopping_rounds': 50,
                           'max_bin': 50,
                           'metric': 'binary_logloss',
                           'verbosity': 1,
                           'seed': train_seed}

        negative_params = {'application': 'binary',
                           'learning_rate': 0.003,
                           'num_leaves': 80,  # <2^(max_depth)
                           'tree_learner': 'serial',
                           'max_depth': 7,  # default=-1
                           'min_data_in_leaf': 2000,  # default=20
                           'feature_fraction': 0.5,  # default=1
                           'bagging_fraction': 0.6,  # default=1
                           'bagging_freq': 5,  # default=0 perform bagging every k iteration
                           'bagging_seed': 1,  # default=3
                           'early_stopping_rounds': 50,
                           'max_bin': 50,
                           'metric': 'binary_logloss',
                           'verbosity': 1,
                           'seed': train_seed}

        models_parameters = [era_training_params, positive_params, negative_params]

        return models_parameters

    @staticmethod
    def train():

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)
        x_train_p, y_train_p, w_train_p, e_train_p, x_g_train_p \
            = utils.load_preprocessed_positive_pd_data(preprocessed_data_path)
        x_train_n, y_train_n, w_train_n, e_train_n, x_g_train_n \
            = utils.load_preprocessed_negative_pd_data(preprocessed_data_path)

        models_parameters = PrejudgeTraining.get_models_parameters()

        positive_era_list = preprocess.positive_era_list
        negative_era_list = preprocess.negative_era_list

        hyper_parameters = {'seed': cv_seed,
                            'n_splits_e': 5,
                            'num_boost_round_e': 8000,
                            'n_cv_e': 10,
                            'n_valid_p': 3,
                            'n_cv_p': 20,
                            'n_era_p': 14,
                            'num_boost_round_p': 500,
                            'era_list_p': positive_era_list,
                            'n_valid_n': 1,
                            'n_cv_n': 6,
                            'n_era_n': 6,
                            'num_boost_round_n': 500,
                            'era_list_n': negative_era_list}

        PES = prejudge.PrejudgeEraSign(x_train, y_train, w_train, e_train, x_g_train,
                                       x_train_p, y_train_p, w_train_p, e_train_p, x_g_train_p,
                                       x_train_n, y_train_n, w_train_n, e_train_n, x_g_train_n,
                                       x_test, id_test, x_g_test)

        PES.train(pred_path=pred_path + 'prejudge/', loss_log_path=loss_log_path + 'prejudge/',
                  negative_era_list=negative_era_list, model_parameters=models_parameters,
                  hyper_parameters=hyper_parameters)


# Stacking
class ModelStacking:

    # Parameters for layer1
    @staticmethod
    def get_layer1_params():

        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',               # gdbt,rf,dart,goss
                      'learning_rate': 0.003,           # default=0.1
                      'num_leaves': 88,                 # default=31     <2^(max_depth)
                      'max_depth': 7,                   # default=-1
                      'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3      reduce over-fit
                      'feature_fraction': 1,            # default=1
                      'feature_fraction_seed': 10,      # default=2
                      'bagging_fraction': 0.8,          # default=1
                      'bagging_freq': 1,                # default=0 perform bagging every k iteration
                      'bagging_seed': 19,               # default=3
                      'lambda_l1': 0,                   # default=0
                      'lambda_l2': 0,                   # default=0
                      'min_gain_to_split': 0,           # default=0
                      'max_bin': 2250,                  # default=255
                      'min_data_in_bin': 5,             # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,      # default=0
                      'seed': train_seed}

        # Parameters of XGBoost
        xgb_params = {'eta': 0.008,
                      'gamma': 0,                       # 如果loss function小于设定值，停止产生子节点
                      'max_depth': 7,                   # default=6
                      'min_child_weight': 15,           # default=1，建立每个模型所需最小样本权重和
                      'subsample': 0.8,                 # 建立树模型时抽取子样本占整个样本的比例
                      'colsample_bytree': 1,            # 建立树时对特征随机采样的比例
                      'colsample_bylevel': 1,
                      'lambda': 0,
                      'alpha': 0,
                      'early_stopping_rounds': 30,
                      'nthread': -1,
                      'objective': 'binary:logistic',
                      'eval_metric': 'logloss',
                      'seed': train_seed}

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
                      'unit_number': [56, 28, 14],
                      'learning_rate': 0.0001,
                      'keep_probability': 0.4,
                      'batch_size': 256,
                      'seed': dnn_seed,
                      'display_step': 100,
                      'save_path': './checkpoints/',
                      'log_path': './log/'}

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

    # Parameters for layer2
    @staticmethod
    def get_layer2_params():

        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',               # gdbt,rf,dart,goss
                      'learning_rate': 0.003,           # default=0.1
                      'num_leaves': 88,                 # default=31     <2^(max_depth)
                      'max_depth': 7,                   # default=-1
                      'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3      reduce over-fit
                      'feature_fraction': 1,            # default=1
                      'feature_fraction_seed': 10,      # default=2
                      'bagging_fraction': 0.8,          # default=1
                      'bagging_freq': 1,                # default=0 perform bagging every k iteration
                      'bagging_seed': 19,               # default=3
                      'lambda_l1': 0,                   # default=0
                      'lambda_l2': 0,                   # default=0
                      'min_gain_to_split': 0,           # default=0
                      'max_bin': 2250,                  # default=255
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
        #               'seed': dnn_seed,
        #               'display_step': 100,
        #               'save_path': './checkpoints/',
        #               'log_path': './log/'}

        # List of parameters for layer2
        layer2_params = [
                        lgb_params,
                        # dnn_params
                        ]

        return layer2_params

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
                      'seed': dnn_seed,
                      'display_step': 100,
                      'save_path': './checkpoints/',
                      'log_path': './log/'}

        # List of parameters for layer3
        layer3_params = [dnn_params]

        return layer3_params

    # Parameters for layer2
    @staticmethod
    def get_final_layer_params():

        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',               # gdbt,rf,dart,goss
                      'learning_rate': 0.003,           # default=0.1
                      'num_leaves': 88,                 # default=31     <2^(max_depth)
                      'max_depth': 7,                   # default=-1
                      'min_data_in_leaf': 2500,         # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,  # default=1e-3      reduce over-fit
                      'feature_fraction': 1,            # default=1
                      'feature_fraction_seed': 10,      # default=2
                      'bagging_fraction': 0.8,          # default=1
                      'bagging_freq': 1,                # default=0 perform bagging every k iteration
                      'bagging_seed': 19,               # default=3
                      'lambda_l1': 0,                   # default=0
                      'lambda_l2': 0,                   # default=0
                      'min_gain_to_split': 0,           # default=0
                      'max_bin': 2250,                  # default=255
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
        #               'seed': dnn_seed,
        #               'display_step': 100,
        #               'save_path': './checkpoints/',
        #               'log_path': './log/'}

        return lgb_params

    @staticmethod
    def deep_stack_train():

        hyper_params = {'n_valid': (4, 4),
                        'n_era': (20, 20),
                        'n_epoch': (3, 1),
                        'cv_seed': cv_seed}

        layer1_prams = ModelStacking.get_layer1_params()
        layer2_prams = ModelStacking.get_layer2_params()
        # layer3_prams = ModelStacking.get_layer3_params()

        layers_params = [layer1_prams,
                         layer2_prams,
                         # layer3_prams
                         ]

        num_boost_round = {'num_boost_round_lgb_l1': 65,
                           'num_boost_round_xgb_l1': 65,
                           'num_boost_round_lgb_l2': 65,
                           'num_boost_round_final': 65}

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        STK = stacking.DeepStack(x_train, y_train, w_train, e_train, x_test, id_test, x_g_train, x_g_test,
                                 pred_path=pred_path + 'stack_results/', loss_log_path=loss_log_path,
                                 stack_output_path=stack_output_path, hyper_params=hyper_params,
                                 layers_params=layers_params, num_boost_round=num_boost_round)

        STK.stack()

    @staticmethod
    def stack_tree_train():

        stack_pred_path = pred_path + 'stack_results/'

        hyper_params = {'n_valid': (4, 4),
                        'n_era': (20, 20),
                        'n_epoch': (1, 8),
                        'cv_seed': cv_seed}

        layer1_params = ModelStacking.get_layer1_params()
        # layer2_params = ModelStacking.get_layer2_params()
        # layer3_params = ModelStacking.get_layer3_params()

        layers_params = [layer1_params,
                         # layer2_params,
                         # layer3_params
                         ]

        num_boost_round = {'num_boost_round_lgb_l1': 65,
                           'num_boost_round_xgb_l1': 30,
                           'num_boost_round_final': 65}

        final_layer_params = ModelStacking.get_final_layer_params()
        final_layer_set = {'model': 'LGB',
                           'n_cv': 20}

        x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_pd_data(preprocessed_data_path)
        x_g_train, x_g_test = utils.load_preprocessed_pd_data_g(preprocessed_data_path)

        STK = stacking.StackTree(x_train, y_train, w_train, e_train, x_test, id_test, x_g_train, x_g_test,
                                 pred_path=stack_pred_path, loss_log_path=loss_log_path,
                                 stack_output_path=stack_output_path, hyper_params=hyper_params,
                                 layers_params=layers_params, num_boost_round=num_boost_round,
                                 final_layer_params=final_layer_params, final_layer_set=final_layer_set)

        STK.stack()


if __name__ == "__main__":

    # Check if directories exit or not
    utils.check_dir(path_list)

    start_time = time.time()

    print('======================================================')
    print('Start Training...')
    print('======================================================')
    # Logistic Regression
    # TrainSingleModel.lr_train()

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
    # GridSearch.stack_lgb_grid_search()

    # Stacking
    # ModelStacking.deep_stack_train()
    # ModelStacking.stack_tree_train()
    TrainSingleModel.stack_lgb_train()

    # Prejudge
    # PrejudgeTraining.train()

    print('======================================================')
    print('All Task Done!')
    print('train_seed: ', train_seed)
    print('cv_seed: ', cv_seed)
    print('dnn_seed: ', dnn_seed)
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')
