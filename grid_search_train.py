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
            ## Auto Grid Search Parameters ##

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
        train_args = {'n_valid': 4,
                      'n_cv': 20,
                      'n_era': 20,
                      'cv_generator': None,
                      'era_list': None,
                      'rescale': False}

        # Training Options
        train_options = {'show_importance': False,
                         'show_accuracy': True,
                         'save_final_pred': True,
                         'save_final_prob_train': False,
                         'save_cv_pred': False,
                         'save_cv_prob_train': False,
                         'save_csv_log': True}

        # Reduced Features
        reduced_feature_list = None

        # Base Parameters
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
        # base_parameters = None

        """
            Auto Grid Search Parameters
        """
        pg_list = [
                   # [['max_depth', [8, 9, 10]], ['min_child_weight', [6, 12, 18]]],
                   [['learning_rate', [0.002, 0.003, 0.005]], ['subsample', [0.8, 0.85, 0.9]]]
                   ]
        TM.auto_grid_search('xgb', parameter_grid_list=pg_list, n_epoch=200,
                            base_parameters=base_parameters, save_final_pred=False,
                            reduced_feature_list=reduced_feature_list, num_boost_round=30,
                            grid_search_n_cv=5, train_args=train_args, train_options=train_options)


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
