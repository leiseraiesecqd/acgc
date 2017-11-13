import random
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

        else:
            base_parameters = None

        return base_parameters

    def train(self):
        """
            ## Train Single Model ##

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
            Global Seed
        """
        train_seed = random.randint(0, 1000)
        cv_seed = random.randint(0, 1000)
        # train_seed = 666
        # cv_seed = 216  # 425 48 461 157

        """
            Training Arguments
        """
        train_args = {'prescale': False,
                      'postscale': True,
                      'use_global_valid': False,
                      'show_importance': False,
                      'show_accuracy': False,
                      'save_final_pred': True,
                      'save_final_prob_train': False,
                      'save_cv_pred': False,
                      'save_cv_prob_train': False,
                      'save_csv_log': True,
                      'append_info': 'fw_v0.2_c20_w35'}

        """
            Cross Validation Arguments
        """
        # cv_args = {'n_valid': 4,
        #            'n_cv': 20,
        #            'n_era': 20}

        cv_args = {'n_valid': 27,
                   'n_cv': 20,
                   'n_era': 135}

        # cv_weights = list(range(1, 21))
        # from math import log
        # cv_weights = [log(i/2 + 1) for i in range(1, 21)]
        # from models.cross_validation import CrossValidation
        # cv_args = {'n_valid': 27,
        #            'valid_rate': 0.2,
        #            'n_cv': 20,
        #            'n_era': 135,
        #            'cv_generator': CrossValidation.forward_window,
        #            'window_size': 35,
        #            # 'cv_generator': CrossValidation.forward_increase,
        #            # 'cv_weights': cv_weights,
        #            }

        """
            Reduced Features
        """
        reduced_feature_list = None

        """
            Base Parameters
        """
        base_parameters = self.get_base_params('lgb')

        # base_parameters = None

        """
            Train Single Model
        """
        TM.train_single_model('lgb', train_seed, cv_seed, num_boost_round=100,
                              reduced_feature_list=reduced_feature_list, base_parameters=base_parameters,
                              train_args=train_args, cv_args=cv_args, use_multi_group=True)

        print('======================================================')
        print('Global Train Seed: {}'.format(train_seed))
        print('Global Cross Validation Seed: {}'.format(cv_seed))


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
