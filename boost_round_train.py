import time
import parameters
from models import utils
from models.training_mode import TrainingMode


class Training:

    def __init__(self):
        pass

    @staticmethod
    def get_base_params(model_name=None):
        """
            Get Base Parameters
        """
        if model_name == 'xgb':
            """
                XGB
            """
            base_parameters = {'learning_rate': 0.003,
                               'gamma': 0.001,
                               'max_depth': 10,
                               'min_child_weight': 8,
                               'subsample': 0.92,
                               'colsample_bytree': 0.85,
                               'colsample_bylevel': 0.7,
                               'lambda': 0,
                               'alpha': 0,
                               'early_stopping_rounds': 10000,
                               'n_jobs': -1,
                               'objective': 'binary:logistic',
                               'eval_metric': 'logloss'}

        elif model_name == 'lgb':
            """
                LGB
            """
            base_parameters = {'application': 'binary',
                               'boosting': 'gbdt',
                               'learning_rate': 0.003,
                               'num_leaves': 88,
                               'max_depth': 9,
                               'min_data_in_leaf': 2500,
                               'min_sum_hessian_in_leaf': 1e-3,
                               'feature_fraction': 0.6,
                               'feature_fraction_seed': 19,
                               'bagging_fraction': 0.8,
                               'bagging_freq': 5,
                               'bagging_seed': 1,
                               'lambda_l1': 0,
                               'lambda_l2': 0,
                               'min_gain_to_split': 0,
                               'max_bin': 225,
                               'min_data_in_bin': 5,
                               'metric': 'binary_logloss',
                               'num_threads': -1,
                               'verbosity': 1,
                               'early_stopping_rounds': 10000}

        elif model_name == 'lgb_fi':
            """
                LGB Forward Increase
            """
            base_parameters = {'application': 'binary',
                               'boosting': 'gbdt',
                               'learning_rate': 0.003,
                               'num_leaves': 88,
                               'max_depth': 8,
                               'min_data_in_leaf': 2500,
                               'min_sum_hessian_in_leaf': 1e-3,
                               'feature_fraction': 0.5,
                               'feature_fraction_seed': 19,
                               'bagging_fraction': 0.7,
                               'bagging_freq': 9,
                               'bagging_seed': 1,
                               'lambda_l1': 0,
                               'lambda_l2': 0,
                               'min_gain_to_split': 0,
                               'max_bin': 225,
                               'min_data_in_bin': 5,
                               'metric': 'binary_logloss',
                               'num_threads': -1,
                               'verbosity': 1,
                               'early_stopping_rounds': 10000}

        elif model_name == 'xgb_fi':
            """
                XGB Forward Increase
            """
            base_parameters = {'learning_rate': 0.003,
                               'gamma': 0.001,
                               'max_depth': 10,
                               'min_child_weight': 8,
                               'subsample': 0.92,
                               'colsample_bytree': 0.85,
                               'colsample_bylevel': 0.7,
                               'lambda': 0,
                               'alpha': 0,
                               'early_stopping_rounds': 10000,
                               'n_jobs': -1,
                               'objective': 'binary:logistic',
                               'eval_metric': 'logloss'
                               }

        elif model_name == 'lgb_fw':
            """
               LGB Forward Window
            """
            base_parameters = {'application': 'binary',
                               'boosting': 'gbdt',
                               'learning_rate': 0.003,
                               'num_leaves': 88,
                               'max_depth': 8,
                               'min_data_in_leaf': 2500,
                               'min_sum_hessian_in_leaf': 1e-3,
                               'feature_fraction': 0.5,
                               'feature_fraction_seed': 19,
                               'bagging_fraction': 0.5,
                               'bagging_freq': 3,
                               'bagging_seed': 1,
                               'lambda_l1': 0,
                               'lambda_l2': 0,
                               'min_gain_to_split': 0,
                               'max_bin': 225,
                               'min_data_in_bin': 5,
                               'metric': 'binary_logloss',
                               'num_threads': -1,
                               'verbosity': 1,
                               'early_stopping_rounds': 10000}

        elif model_name == 'xgb_fw':
            """
                XGB Forward Window
            """
            base_parameters = {'learning_rate': 0.003,
                               'gamma': 0.001,
                               'max_depth': 10,
                               'min_child_weight': 8,
                               'subsample': 0.92,
                               'colsample_bytree': 0.85,
                               'colsample_bylevel': 0.7,
                               'lambda': 0,
                               'alpha': 0,
                               'early_stopping_rounds': 10000,
                               'n_jobs': -1,
                               'objective': 'binary:logistic',
                               'eval_metric': 'logloss'}

        else:
            print('------------------------------------------------------')
            print('[W] Training without Base Parameters')
            base_parameters = None

        return base_parameters

    def get_cv_args(self, model_name=None):

        from models.cross_validation import CrossValidation

        if model_name == 'lgb_fi':
            """
                LGB Forward Increase
            """
            cv_weights = self.get_cv_weight('range', 1, 21)
            cv_args = {'valid_rate': 0.125,
                       'n_cv': 20,
                       'n_era': 119,
                       'cv_generator': CrossValidation.forward_increase,
                       'cv_weights': cv_weights}

        elif model_name == 'xgb_fi':
            """
                XGB Forward Increase
            """
            cv_weights = self.get_cv_weight('range', 1, 21)
            cv_args = {'valid_rate': 0.125,
                       'n_cv': 20,
                       'n_era': 119,
                       'cv_generator': CrossValidation.forward_increase,
                       'cv_weights': cv_weights}

        elif model_name == 'lgb_fw':
            """
               LGB Forward Window
            """
            cv_args = {'valid_rate': 0.1,
                       'n_cv': 10,
                       'n_era': 119,
                       'cv_generator': CrossValidation.forward_window,
                       'window_size': 40}

        elif model_name == 'xgb_fw':
            """
                XGB Forward Window
            """
            cv_args = {'valid_rate': 0.1,
                       'n_cv': 10,
                       'n_era': 119,
                       'cv_generator': CrossValidation.forward_window,
                       'window_size': 40}

        else:
            cv_args = {'n_valid': 27,
                       'n_cv': 20,
                       'n_era': 119}
            print('------------------------------------------------------')
            print('[W] Training with Base cv_args:\n', cv_args)

        return cv_args

    @staticmethod
    def get_cv_weight(mode, start, stop):

        from math import log

        if mode == 'range':
            cv_weights = list(range(start, stop))
        elif mode == 'log':
            cv_weights = [log(i/2 + 1) for i in range(start, stop)]
        else:
            cv_weights = [1 for _ in range(start, stop)]

        return cv_weights

    def train(self):
        """
            ## Auto Train with Logs of Boost Round ##

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
            'christar':     Christar1991
            'prejudge_b':   PrejudgeBinary
            'prejudge_m':   PrejudgeMultiClass
            'stack_t':      StackTree
        """
        TM = TrainingMode()

        """
            Training Arguments
        """
        train_args = {'prescale': False,
                      'postscale': True,
                      'use_global_valid': False,
                      'use_custom_obj': False,
                      'show_importance': False,
                      'show_accuracy': False,
                      'save_final_pred': True,
                      'save_final_prob_train': False,
                      'save_cv_pred': False,
                      'save_cv_prob_train': False,
                      'save_csv_log': True,
                      'append_info': 'forward_increase_postscale'}

        """
            Cross Validation Arguments
        """
        # cv_args = {'n_valid': 4,
        #            'n_cv': 20,
        #            'n_era': 20}

        cv_args = self.get_cv_args('xgb_fi')

        """
            Reduced Features
        """
        reduced_feature_list = None

        """
            Base Parameters
        """
        base_parameters = self.get_base_params('xgb_fi')

        # base_parameters = None

        """
            Auto Train with Logs of Boost Round
        """
        # cv_weights_range = [self.get_cv_weight('range', 1, i+1) for i in [5, 8, 10, 12, 15, 20]]
        # cv_weights_log = [self.get_cv_weight('log', 1, i+1) for i in [5, 8, 10, 12, 15, 20]]
        # n_cv_list = [5, 8, 10, 12, 15, 20] * 6
        # import numpy as np
        # valid_rate_list = np.array([[i]*6 for i in [0.075, 0.1, 0.125, 0.15, 0.175, 0.2]]).reshape(-1,).tolist()
        # cv_weights_list = cv_weights_log*6
        pg_list = [
                   # [['n_cv', (5, 10, 12, 15, 20)],
                   #  ['valid_rate', (0.1, 0.125, 0.166, 0.2, 0.25)],
                   #  ['window_size', (32, 35, 40, 48, 54, 60)]]
                   # [['n_cv', n_cv_list],
                   #  ['valid_rate', valid_rate_list],
                   #  ['cv_weights', cv_weights_list]]
                   # [['learning_rate', [0.003]]]
                   # [['max_depth', (7, 8, 9, 10, 11, 12)]],
                   # [['feature_fraction' (0.5, 0.6, 0.7, 0.8, 0.9)]],
                   # [['bagging_fraction', (0.5, 0.6, 0.7, 0.8, 0.9)]],
                   # [['bagging_freq', (1, 3, 5, 7, 9, 11)]]
                   [
                    ['max_depth', (8, 9, 10)],
                    ['min_child_weight', (2, 4, 6)],
                    ['subsample', (0.8, 0.85, 0.9)],
                    ['colsample_bytree', (0.8, 0.85, 0.9)],
                    ['colsample_bylevel', (0.7, 0.75, 0.8)]
                    ]
                   ]
        train_seed_list = [999]
        cv_seed_list = [95]
        # train_seed_list = None
        # cv_seed_list = None
        TM.auto_train_boost_round('xgb', num_boost_round=100, n_epoch=1, full_grid_search=True,
                                  use_multi_group=False, train_seed_list=train_seed_list, cv_seed_list=cv_seed_list,
                                  base_parameters=base_parameters, parameter_grid_list=pg_list, save_final_pred=True,
                                  reduced_feature_list=reduced_feature_list, train_args=train_args, cv_args=cv_args)


if __name__ == "__main__":

    start_time = time.time()

    # Check if directories exit or not
    utils.check_dir(parameters.path_list)

    print('======================================================')
    print('Start Training...')

    T = Training()
    T.train()

    print('------------------------------------------------------')
    print('All Tasks Done!')
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')
