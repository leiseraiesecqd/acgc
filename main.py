import random
import time
import parameters
from models import utils
from models.training_mode import TrainingMode
from models.cross_validation import CrossValidation


class Training:

    def __init__(self):
        pass

    @staticmethod
    def train():
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
            'christar':     Christar1991
            'prejudge_b':   PrejudgeBinary
            'prejudge_m':   PrejudgeMultiClass
            'stack_t':      StackTree
        """
        TM = TrainingMode()

        """
            Global Seed
        """
        train_seed = random.randint(0, 500)
        cv_seed = random.randint(0, 500)
        # train_seed = 666
        # cv_seed = 216  # 425 48 461 157

        """
            Training Arguments
        """
        train_args = {'train_seed': train_seed,
                      'rescale': False,
                      'show_importance': False,
                      'show_accuracy': True,
                      'save_final_pred': True,
                      'save_final_prob_train': False,
                      'save_cv_pred': False,
                      'save_cv_prob_train': False,
                      'save_csv_log': True,
                      'append_info': None}

        """
            Cross Validation Arguments
        """
        cv_args = {'n_valid': 4,
                   'n_cv': 20,
                   'n_era': 20,
                   'cv_seed': cv_seed,
                   'cv_generator': None}

        # cv_args = {'n_valid': 27,
        #            'n_cv': 20,
        #            'n_era': 135,
        #            'cv_seed': cv_seed,
        #            'cv_generator': None}

        # cv_args = {'n_valid': 27,
        #            'n_cv': 20,
        #            'n_era': 135,
        #            'cv_seed': cv_seed,
        #            # 'cv_generator': CrossValidation.forward_window_validation,
        #            # 'window_size': 35,
        #            # 'cv_generator': CrossValidation.forward_increase_validation,
        #            # 'valid_rate': 0.2
        #            }

        """
            Reduced Features
        """
        reduced_feature_list = None

        """
            Base Parameters
        """
        """ XGB """
        base_parameters = {'learning_rate': 0.003,
                           'gamma': 0.001,              # 如果loss function小于设定值，停止产生子节点
                           'max_depth': 10,             # default=6
                           'min_child_weight': 12,      # default=1，建立每个模型所需最小样本权重和
                           'subsample': 0.92,           # 建立树模型时抽取子样本占整个样本的比例
                           'colsample_bytree': 0.85,    # 建立树时对特征随机采样的比例
                           'colsample_bylevel': 0.7,
                           'lambda': 0,
                           'alpha': 0,
                           'early_stopping_rounds': 10000,
                           'n_jobs': -1,
                           'objective': 'binary:logistic',
                           'eval_metric': 'logloss'}

        """ LGB """
        # base_parameters = {'application': 'binary',
        #                    'boosting': 'gbdt',                  # gdbt,rf,dart,goss
        #                    'learning_rate': 0.003,              # default=0.1
        #                    'num_leaves': 88,                    # default=31       <2^(max_depth)
        #                    'max_depth': 9,                      # default=-1
        #                    'min_data_in_leaf': 2500,            # default=20       reduce over-fit
        #                    'min_sum_hessian_in_leaf': 1e-3,     # default=1e-3     reduce over-fit
        #                    'feature_fraction': 0.5,             # default=1
        #                    'feature_fraction_seed': 19,         # default=2
        #                    'bagging_fraction': 0.8,             # default=1
        #                    'bagging_freq': 2,                   # default=0        perform bagging every k iteration
        #                    'bagging_seed': 1,                   # default=3
        #                    'lambda_l1': 0,                      # default=0
        #                    'lambda_l2': 0,                      # default=0
        #                    'min_gain_to_split': 0,              # default=0
        #                    'max_bin': 225,                      # default=255
        #                    'min_data_in_bin': 5,                # default=5
        #                    'metric': 'binary_logloss',
        #                    'num_threads': -1,
        #                    'verbosity': 1,
        #                    'early_stopping_rounds': 10000}

        # base_parameters = None

        """
            Train Single Model
        """
        # TM.train_single_model('xgb', train_seed, cv_seed, num_boost_round=88,
        #                       reduced_feature_list=reduced_feature_list, base_parameters=base_parameters,
        #                       train_args=train_args, use_multi_group=False)

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
        TM.auto_train_boost_round('xgb', train_seed_list, cv_seed_list, n_epoch=1, base_parameters=base_parameters,
                                  num_boost_round=115, parameter_grid_list=pg_list, save_final_pred=True,
                                  reduced_feature_list=reduced_feature_list, grid_search_n_cv=20,
                                  train_args=train_args, cv_args=cv_args, use_multi_group=False)

        """
            Auto Grid Search Parameters
        """
        # pg_list = [
        #            # [['max_depth', [8, 9, 10]], ['min_child_weight', [6, 12, 18]]],
        #            [['learning_rate', [0.002, 0.003, 0.005]], ['subsample', [0.8, 0.85, 0.9]]]
        #            ]
        # TM.auto_grid_search('lgb', parameter_grid_list=pg_list, n_epoch=200, base_parameters=base_parameters,
        #                     save_final_pred=False, reduced_feature_list=reduced_feature_list, num_boost_round=30,
        #                     grid_search_n_cv=5, train_args=train_args, use_multi_group=False)

        """
            Auto Train
        """
        # TM.auto_train('lgb', n_epoch=10000, base_parameters=base_parameters,
        #               reduced_feature_list=reduced_feature_list, train_args=train_args, use_multi_group=False)

        """
            Others
        """
        # TM.train_single_model('dnn', train_seed, cv_seed,  reduced_feature_list=reduced_feature_list,
        #                       base_parameters=base_parameters, train_args=train_args,  use_multi_group=False)
        # TM.train_single_model('prejudge_b', train_seed, cv_seed, load_pickle=False,
        #                       base_parameters=base_parameters, reduced_feature_list=reduced_feature_list,
        #                       train_args=train_args, use_multi_group=False)
        # TM.train_single_model('stack_lgb', train_seed, cv_seed, auto_idx=1,
        #                       base_parameters=base_parameters, reduced_feature_list=reduced_feature_list,
        #                       train_args=train_args, use_multi_group=False)
        # TM.train_single_model('stack_pt', train_seed, cv_seed, reduced_feature_list=reduced_feature_list,
        #                       base_parameters=base_parameters, train_args=train_args, use_multi_group=False)

        # pg_list = [
        #            [['learning_rate', [0.00005]]],
        #            [['keep_probability', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]],
        #            # [['unit_number',
        #            #   [
        #            #    [32, 16, 8],
        #            #    [48, 24, 12],
        #            #    [64, 32], [64, 32, 16],
        #            #    [128, 64], [128, 64, 32], [128, 64, 32, 16],
        #            #    [256, 128], [256, 128, 64], [256, 128, 64, 32], [256, 128, 64, 32, 16],
        #            #    [200, 100, 50],
        #            #    [2048, 512],
        #            #    [288, 144, 72], [288, 144, 72, 36],
        #            #    [216, 108, 54], [216, 108, 54, 27],
        #            #    [128, 256, 128, 64], [64, 128, 64, 32], [128, 256, 128], [64, 128, 64]
        #            #    ]]]
        #            ]
        # train_seed_list = [666]
        # cv_seed_list = [216]
        # TM.auto_train_boost_round('dnn', train_seed_list, cv_seed_list, n_epoch=1, base_parameters=base_parameters,
        #                           epochs=2, parameter_grid_list=pg_list, save_final_pred=True,
        #                           reduced_feature_list=reduced_feature_list, grid_search_n_cv=20,
        #                           train_args=train_args, use_multi_group=False)

        # TM.auto_train('stack_t', n_epoch=2, stack_final_epochs=10, base_parameters=base_parameters,
        #               reduced_feature_list=reduced_feature_list, train_args=train_args, use_multi_group=False)

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
