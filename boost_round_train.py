import time
import parameters
from models import utils
from models.training_mode import TrainingMode


class Training:

    def __init__(self):
        pass

    @staticmethod
    def train():
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

        # Training Arguments
        train_args = {'n_valid': 27,
                      'n_cv': 20,
                      'n_era': 135,
                      'cv_generator': None,
                      'era_list': None,
                      'rescale': False,
                      'show_importance': False,
                      'show_accuracy': False,
                      'save_final_pred': True,
                      'save_final_prob_train': False,
                      'save_cv_pred': False,
                      'save_cv_prob_train': False,
                      'save_csv_log': True,
                      'append_info': '5-fold'}

        # Reduced Features
        reduced_feature_list = None

        # Base Parameters
        """ XGB """
        base_parameters = {'learning_rate': 0.003,
                           'gamma': 0.001,
                           'max_depth': 10,
                           'min_child_weight': 6,
                           'subsample': 0.88,
                           'colsample_bytree': 0.85,
                           'colsample_bylevel': 0.75,
                           'lambda': 0,
                           'alpha': 0,
                           'early_stopping_rounds': 10000,
                           'n_jobs': -1,
                           'objective': 'binary:logistic',
                           'eval_metric': 'logloss'}

        """ LGB """
        # base_parameters = {'application': 'binary',
        #                    'boosting': 'gbdt',
        #                    'learning_rate': 0.003,
        #                    'num_leaves': 88,
        #                    'max_depth': 9,
        #                    'min_data_in_leaf': 2500,
        #                    'min_sum_hessian_in_leaf': 1e-3,
        #                    'feature_fraction': 0.5,
        #                    'feature_fraction_seed': 19,
        #                    'bagging_fraction': 0.8,
        #                    'bagging_freq': 2,
        #                    'bagging_seed': 1,
        #                    'lambda_l1': 0,
        #                    'lambda_l2': 0,
        #                    'min_gain_to_split': 0,
        #                    'max_bin': 225,
        #                    'min_data_in_bin': 5,
        #                    'metric': 'binary_logloss',
        #                    'num_threads': -1,
        #                    'verbosity': 1,
        #                    'early_stopping_rounds': 10000}

        # base_parameters = None

        """
            Auto Train with Logs of Boost Round
        """
        pg_list = [
                   [['learning_rate', [0.003]]]
                   ]
        # train_seed_list = [666]
        # cv_seed_list = [216]
        train_seed_list = None
        cv_seed_list = None
        TM.auto_train_boost_round('xgb', train_seed_list, cv_seed_list, n_epoch=200, base_parameters=base_parameters,
                                  num_boost_round=135, parameter_grid_list=pg_list, save_final_pred=True,
                                  reduced_feature_list=reduced_feature_list, grid_search_n_cv=20,
                                  train_args=train_args, use_multi_group=False)


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
