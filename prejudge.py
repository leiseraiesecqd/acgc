import models
import utils
import time
import numpy as np


# Pre Judge Era Sign
class PrejudgeEraSign:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_g_tr, x_tr_p, y_tr_p, w_tr_p, e_tr_p, x_g_tr_p,
                 x_tr_n, y_tr_n, w_tr_n, e_tr_n, x_g_tr_n, x_te, id_te, x_g_te,):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_g_train = x_g_tr
        self.x_train_p = x_tr_p
        self.y_train_p = y_tr_p
        self.w_train_p = w_tr_p
        self.e_train_p = e_tr_p
        self.x_g_train_p = x_g_tr_p
        self.x_train_n = x_tr_n
        self.y_train_n = y_tr_n
        self.w_train_n = w_tr_n
        self.e_train_n = e_tr_n
        self.x_g_train_n = x_g_tr_n
        self.x_test = x_te
        self.id_test = id_te
        self.x_g_test = x_g_te

    def predict_era_sign(self, pred_path, negative_era_list, n_splits_e, n_cv_e, cv_seed,
                         use_weight=True, force_convert_era=True, parameters_e=None):

        print('======================================================')
        print('Training Era Sign...')

        feature = self.x_g_train
        # Convert Eras of Training Data to 0 and 1
        era_sign_train = np.array([0 if era in negative_era_list else 1 for era in self.e_train])

        # Init Model
        LGB = models.LightGBM(self.x_train, era_sign_train, self.w_train, self.e_train,
                              self.x_test, self.id_test, feature, self.x_g_test)

        # Training and Get Probabilities of Test Era Being Positive
        era_prob_test = LGB.era_train(pred_path + 'pred_era/', n_splits=n_splits_e, n_cv=n_cv_e,
                                      cv_seed=cv_seed, use_weight=use_weight, parameters=parameters_e)

        # Convert Probabilities of Test Era to 0 and 1
        if force_convert_era is True:
            era_sign_test = np.array([0 if era_prob < 0.5 else 1 for era_prob in era_prob_test])
        else:
            era_sign_test = era_prob_test

        return era_sign_test

    def split_data_by_era_sign(self, era_sign_test, seed):

        if seed is not None:
            np.random.seed(seed)

        era_idx_test_p = []
        era_idx_test_n = []

        for idx, era_sign in enumerate(era_sign_test):

            if era_sign == 1:
                era_idx_test_p.append(idx)
            else:
                era_idx_test_n.append(idx)

        np.random.shuffle(era_idx_test_p)
        np.random.shuffle(era_idx_test_n)

        # Split Set of Test Set
        x_test_p = self.x_test[era_idx_test_p]
        x_g_test_p = self.x_test[era_idx_test_p]
        x_test_n = self.x_test[era_idx_test_n]
        x_g_test_n = self.x_test[era_idx_test_n]

        return x_test_p, x_g_test_p, era_idx_test_p, x_test_n, x_g_test_n, era_idx_test_n

    def train_models_by_era_sign(self, x_test_p, x_g_test_p, era_idx_test_p, x_test_n, x_g_test_n, era_idx_test_n,
                                 pred_path, loss_log_path, cv_seed, n_valid_p, n_cv_p, n_era_p, parameters_p,
                                 n_valid_n, n_cv_n, n_era_n, parameters_n):

        print('======================================================')
        print('Training Models by Era Sign...')

        LGBM_P = models.LightGBM(self.x_train_p, self.y_train_p, self.w_train_p, self.e_train_p,
                                 x_test_p, self.id_test, self.x_g_train_p, x_g_test_p)

        LGBM_N = models.LightGBM(self.x_train_n, self.y_train_n, self.w_train_n, self.e_train_n,
                                 x_test_n, self.id_test, self.x_g_train_n, x_g_test_n)

        print('======================================================')
        print('Training Models of Positive Era Sign...')

        prob_test_p = LGBM_P.train(pred_path + 'positive/', loss_log_path + 'positive/', n_valid=n_valid_p,
                                   n_cv=n_cv_p, n_era=n_era_p,  cv_seed=cv_seed, parameters=parameters_p,
                                   return_prob_test=True)

        print('======================================================')
        print('Training Models of Negative Era Sign...')

        prob_test_n = LGBM_N.train(pred_path + 'negative/', loss_log_path + 'negative/', n_valid=n_valid_n,
                                   n_cv=n_cv_n, n_era=n_era_n, cv_seed=cv_seed, parameters=parameters_n,
                                   return_prob_test=True)

        prob_test = np.zeros_like(self.id_test, dtype=np.float64)

        for idx_p, prob_p in zip(era_idx_test_p, prob_test_p):
            prob_test[idx_p] = prob_p

        for idx_n, prob_n in zip(era_idx_test_n, prob_test_n):
            prob_test[idx_n] = prob_n

        return prob_test

    def train(self, pred_path, loss_log_path, negative_era_list, model_parameters, hyper_parameters):

        start_time = time.time()

        path_list = [pred_path,
                     pred_path + 'positive/',
                     pred_path + 'negative/',
                     pred_path + 'pred_era/',
                     pred_path + 'pred_era/final_results/',
                     pred_path + 'final_results/',
                     pred_path + 'era_sign_test_pickle/',
                     loss_log_path,
                     loss_log_path + 'positive/',
                     loss_log_path + 'negative/']
        utils.check_dir(path_list)

        parameters_e = model_parameters[0]
        parameters_p = model_parameters[1]
        parameters_n = model_parameters[2]

        seed = hyper_parameters['seed']
        n_splits_e = hyper_parameters['n_splits_e']
        n_cv_e = hyper_parameters['n_cv_e']
        n_valid_p = hyper_parameters['n_valid_p']
        n_cv_p = hyper_parameters['n_cv_p']
        n_era_p = hyper_parameters['n_era_p']
        n_valid_n = hyper_parameters['n_valid_n']
        n_cv_n = hyper_parameters['n_cv_n']
        n_era_n = hyper_parameters['n_era_n']

        print('======================================================')
        print('Start training...')

        # Training Era Sign
        era_sign_test = self.predict_era_sign(pred_path, negative_era_list, n_splits_e, n_cv_e, seed,
                                              use_weight=False,  force_convert_era=True, parameters_e=parameters_e)

        # Save era_sign_test to Pickle File
        utils.save_np_to_pkl(era_sign_test, pred_path + 'era_sign_test_pickle/')

        x_test_p, x_g_test_p, era_idx_test_p, x_test_n, \
            x_g_test_n, era_idx_test_n = self.split_data_by_era_sign(era_sign_test, seed)

        # Training Models by Era Sign
        prob_test = \
            self.train_models_by_era_sign(x_test_p, x_g_test_p, era_idx_test_p, x_test_n, x_g_test_n,  era_idx_test_n,
                                          pred_path=pred_path, loss_log_path=loss_log_path,
                                          cv_seed=seed, n_valid_p=n_valid_p, n_cv_p=n_cv_p, n_era_p=n_era_p,
                                          parameters_p=parameters_p, n_valid_n=n_valid_n, n_cv_n=n_cv_n,
                                          n_era_n=n_era_n,  parameters_n=parameters_n)

        # Save Predictions
        utils.save_pred_to_csv(pred_path + 'final_results/', self.id_test, prob_test)

        total_time = time.time() - start_time
        print('======================================================')
        print('Training Done!')
        print('Total Time: {}s'.format(total_time))
        print('======================================================')


if __name__ == '__main__':

    pass
