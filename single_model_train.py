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

        # Create Global Seed for Training and Cross Validation
        train_seed = random.randint(0, 500)
        cv_seed = random.randint(0, 500)
        # train_seed = 666
        # cv_seed = 216  # 425 48 461 157

        # Training Arguments
        train_args = {'n_valid': 7,  # 27,
                      'n_cv': 10,  # 20,
                      'n_era': 135,
                      'window_size': 35,
                      'train_seed': train_seed,
                      'cv_seed': cv_seed,
                      # 'cv_generator': None,
                      'cv_generator': CrossValidation.forward_window_validation,
                      'era_list': None,
                      'rescale': True,
                      'show_importance': False,
                      'show_accuracy': False,
                      'save_final_pred': True,
                      'save_final_prob_train': False,
                      'save_cv_pred': False,
                      'save_cv_prob_train': False,
                      'save_csv_log': True,
                      'append_info': None}

        # Reduced Features
        reduced_feature_list = None

        # Base Parameters
        # base_parameters = {'learning_rate': 0.003,
        #                    'gamma': 0.001,              # 如果loss function小于设定值，停止产生子节点
        #                    'max_depth': 10,             # default=6
        #                    'min_child_weight': 12,      # default=1，建立每个模型所需最小样本权重和
        #                    'subsample': 0.92,           # 建立树模型时抽取子样本占整个样本的比例
        #                    'colsample_bytree': 0.85,    # 建立树时对特征随机采样的比例
        #                    'colsample_bylevel': 0.7,
        #                    'lambda': 0,
        #                    'alpha': 0,
        #                    'early_stopping_rounds': 10000,
        #                    'n_jobs': -1,
        #                    'objective': 'binary:logistic',
        #                    'eval_metric': 'logloss',
        #                    'seed': train_seed}
        base_parameters = None

        """
            Train Single Model
        """
        TM.train_single_model('lgb', train_seed, cv_seed, num_boost_round=100,
                              reduced_feature_list=reduced_feature_list, base_parameters=base_parameters,
                              train_args=train_args, use_multi_group=True)

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
