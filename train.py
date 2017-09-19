import preprocess
import utils
import model
import time

preprocessed_path = './preprocessed_data/'

# HyperParameters
# hyper_parameters = {'version': '1.0',
#                     'epochs': 10,
#                     'layers_number': 10,
#                     'unit_number': [200, 400, 800, 800, 800, 800, 800, 800, 400, 200],
#                     'learning_rate': 0.01,
#                     'keep_probability': 0.75,
#                     'batch_size': 512,
#                     'display_step': 100,
#                     'save_path': './checkpoints/',
#                     'log_path': './log/'}
#
# pickled_data_path = './preprocessed_data/'
#
# print('Loading data set...')
# tr, tr_y, tr_w, val_x, val_y, val_w = model.load_data(pickled_data_path)
#
# dnn = model.DNN(tr, tr_y, tr_w, val_x, val_y, val_w, hyper_parameters)
# dnn.train()
#
# print('Done!')

if __name__ == "__main__":

    start_time = time.time()

    train_x, train_y, train_w = utils.load_data(preprocessed_path)

    xgb_parameters = {'base_score': 0.5,
                      'colsample_bylevel': 1,
                      'colsample_bytree': 0.8,
                      'gamma': 2,
                      'learning_rate': 0.05,
                      'max_delta_step': 0,
                      'max_depth': 3,
                      'min_child_weight': 1,
                      'missing': None,
                      'n_estimators': 100,
                      'nthread': -1,
                      'objective': 'binary:logistic',
                      'reg_alpha': 0,
                      'reg_lambda': 1,
                      'scale_pos_weight': 1,
                      'seed': 0,
                      'silent': 1,
                      'eval_metric': 'logloss',
                      'subsample': 0.8}

    XGB = model.XGBoost(train_x, train_y, train_w)

    clf_xgb = XGB.clf(xgb_parameters)

    parameters_grid = {'base_score': (0.5),
                       # 'colsample_bylevel': 1,
                       'colsample_bytree': (0.5, 0.8, 1.0),
                       # 'gamma': 2,
                       'learning_rate': (0.01, 0.05, 0.1, 0.15, 0.2),
                       # 'max_delta_step': 0,
                       'max_depth': (4, 6, 8, 10),
                       'min_child_weight': (0.8, 1.0, 1.2),
                       # 'missing': None,
                       # 'n_estimators': 100,
                       # 'nthread': -1,
                       # 'objective': 'binary:logistic',
                       # 'reg_alpha': 0,
                       # 'reg_lambda': 1,
                       # 'scale_pos_weight': 1,
                       # 'seed': 1,
                       # 'silent': True,
                       # 'eval_metric': 'logloss',
                       'subsample': (0.5, 0.8, 1.0)}

    model.grid_search(train_x, train_y, clf_xgb, parameters_grid)

    total_time = time.time() - start_time

    print('Done!')
    print('Using {:.3}s'.format(total_time))


