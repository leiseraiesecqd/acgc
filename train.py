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

    XGB = model.XGBoost(train_x, train_y, train_w)

    clf_xgb = XGB.clf()

    parameters = {'base_score': 0.5,
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
                  'silent': True,
                  'subsample': 0.8}

    model.grid_search(train_x, train_y, clf_xgb, parameters)

    total_time = time.time() - start_time

    print('Done!')
    print('Using {:.3}s'.format(total_time))


