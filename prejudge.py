import models
import utils
import time
import numpy as np


class PrejudgeEraSign:
    """
        Prejudge - Training by Split Era sign
    """

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_g_tr, x_tr_p, y_tr_p, w_tr_p, e_tr_p, x_g_tr_p,
                 x_tr_n, y_tr_n, w_tr_n, e_tr_n, x_g_tr_n, x_te, id_te, x_g_te,
                 pred_path, prejudged_data_path, loss_log_path, models_parameters, hyper_parameters):

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
        self.pred_path = pred_path
        self.prejudged_data_path = prejudged_data_path
        self.loss_log_path = loss_log_path
        self.parameters_e = models_parameters[0]
        self.parameters_p = models_parameters[1]
        self.parameters_n = models_parameters[2]
        self.cv_seed = hyper_parameters['cv_seed']
        self.train_seed = hyper_parameters['train_seed']
        self.n_splits_e = hyper_parameters['n_splits_e']
        self.num_boost_round_e = hyper_parameters['num_boost_round_e']
        self.n_cv_e = hyper_parameters['n_cv_e']
        self.n_valid_p = hyper_parameters['n_valid_p']
        self.n_cv_p = hyper_parameters['n_cv_p']
        self.n_era_p = hyper_parameters['n_era_p']
        self.num_boost_round_p = hyper_parameters['num_boost_round_p']
        self.era_list_p = hyper_parameters['era_list_p']
        self.n_valid_n = hyper_parameters['n_valid_n']
        self.n_cv_n = hyper_parameters['n_cv_n']
        self.n_era_n = hyper_parameters['n_era_n']
        self.num_boost_round_n = hyper_parameters['num_boost_round_n']
        self.era_list_n = hyper_parameters['era_list_n']
        self.force_convert_era = hyper_parameters['force_convert_era']
        self.use_weight = hyper_parameters['use_weight']
        self.show_importance = hyper_parameters['show_importance']
        self.show_accuracy = hyper_parameters['show_accuracy']

    @staticmethod
    def load_era_sign_csv(pred_path):
        """
            Load era sign from existed csv file
        """

        f = np.loadtxt(pred_path, dtype=np.float64, skiprows=1, delimiter=",")
        era_prob_test = f[:, -1]

        era_sign_test = np.array([0 if era_prob < 0.5 else 1 for era_prob in era_prob_test])

        return era_sign_test

    def era_prejudge_model_initializer(self, x_train, x_g_train, era_sign_train):
        """
            Initialize model for era prejudging
        """

        LGB_E = models.LightGBM(x_g_train, era_sign_train, self.w_train, self.e_train,
                                self.x_g_test, self.id_test, num_boost_round=self.num_boost_round_e)
        # XGB_E = models.XGBoost(x_train, era_sign_train, self.w_train, self.e_train,
        #                        self.x_test, self.id_test, num_boost_round=self.num_boost_round_e)
        # DNN_E = models.DeepNeuralNetworks(x_train, era_sign_train, self.w_train,
        #                                   self.e_train, self.x_test, self.id_test, self.dnn_l1_params)

        model_e = LGB_E

        return model_e

    def positive_model_initializer(self, x_test_p, x_g_test_p, id_test_p):
        """
            Initialize model for positive eras
        """

        LGB_P = models.LightGBM(self.x_g_train_p, self.y_train_p, self.w_train_p, self.e_train_p,
                                x_g_test_p, id_test_p, num_boost_round=self.num_boost_round_p)
        # XGB_P = models.XGBoost(self.x_train_p, self.y_train_p, self.w_train_p, self.e_train_p,
        #                        x_test_p, id_test_p, num_boost_round=self.num_boost_round_p)
        # DNN_P = models.DeepNeuralNetworks(self.x_train_p, self.y_train_p, self.w_train_p,
        #                                   self.e_train_p, x_test_p, id_test_p, self.dnn_l1_params)

        model_p = LGB_P

        return model_p

    def negative_model_initializer(self, x_test_n, x_g_test_n, id_test_n):
        """
            Initialize model for negative eras
        """

        LGB_N = models.LightGBM(self.x_g_train_n, self.y_train_n, self.w_train_n, self.e_train_n,
                                x_g_test_n, id_test_n, num_boost_round=self.num_boost_round_n)
        # XGB_N = models.XGBoost(self.x_train_n, self.y_train_n, self.w_train_n, self.e_train_n,
        #                        x_test_n, id_test_n, num_boost_round=self.num_boost_round_n)
        # DNN_N = models.DeepNeuralNetworks(self.x_train_n, self.y_train_n, self.w_train_n,
        #                                   self.e_train_n, x_test_n, id_test_n, self.dnn_l1_params)

        model_n = LGB_N

        return model_n

    def predict_era_sign(self):
        """
            Training and predict era signs of instances
        """
        print('======================================================')
        print('Positive Era List: {}'.format(self.era_list_p))
        print('Negative Era List: {}'.format(self.era_list_n))
        print('======================================================')
        print('Training Era Sign...')

        # Convert Eras of Training Data to 0 and 1
        era_sign_train = np.array([0 if era in self.era_list_n else 1 for era in self.e_train])

        # Init Model
        print('------------------------------------------------------')
        print('Initializing Model...')
        model = self.era_prejudge_model_initializer(self.x_train, self.x_g_train, era_sign_train)

        # Training and Get Probabilities of Test Era Being Positive
        era_prob_test = model.prejudge_train(self.pred_path + 'pred_era/', n_splits=self.n_splits_e,
                                             n_cv=self.n_cv_e, cv_seed=self.cv_seed, use_weight=self.use_weight,
                                             parameters=self.parameters_e, show_importance=self.show_importance,
                                             show_accuracy=self.show_accuracy)

        # Convert Probabilities of Test Era to 0 and 1
        if self.force_convert_era is True:
            era_sign_test = np.array([0 if era_prob < 0.5 else 1 for era_prob in era_prob_test])
        else:
            era_sign_test = era_prob_test

        return era_sign_test

    def split_data_by_era_sign(self, era_sign_test):
        """
            Split whole data set to positive and negative set using era sign
        """

        if self.cv_seed is not None:
            np.random.seed(self.cv_seed)

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
        id_test_p = self.id_test[era_idx_test_p]
        x_test_n = self.x_test[era_idx_test_n]
        x_g_test_n = self.x_test[era_idx_test_n]
        id_test_n = self.id_test[era_idx_test_n]

        return x_test_p, x_g_test_p, id_test_p, era_idx_test_p, x_test_n, x_g_test_n, id_test_n, era_idx_test_n

    def train_models_by_era_sign(self, x_test_p, x_g_test_p, id_test_p, era_idx_test_p,
                                 x_test_n, x_g_test_n, id_test_n, era_idx_test_n):
        """
            Training positive and negative model using data set respectively
        """

        print('======================================================')
        print('Training Models by Era Sign...')

        print('------------------------------------------------------')
        print('Initializing Model...')
        model_p = self.positive_model_initializer(x_test_p, x_g_test_p, id_test_p)
        model_n = self.positive_model_initializer(x_test_n, x_g_test_n, id_test_n)

        print('======================================================')
        print('Training Models of Positive Era Sign...')
        prob_test_p = model_p.train(self.pred_path + 'positive/', self.loss_log_path + 'positive/',
                                    n_valid=self.n_valid_p, n_cv=self.n_cv_p, n_era=self.n_era_p,
                                    train_seed=self.train_seed, cv_seed=self.cv_seed, parameters=self.parameters_p,
                                    return_prob_test=True, era_list=self.era_list_p,
                                    show_importance=self.show_importance, show_accuracy=self.show_accuracy,
                                    save_csv_log=True, csv_idx='prejudge_p')

        print('======================================================')
        print('Training Models of Negative Era Sign...')
        prob_test_n = model_n.train(self.pred_path + 'negative/', self.loss_log_path + 'negative/',
                                    n_valid=self.n_valid_n, n_cv=self.n_cv_n, n_era=self.n_era_n,
                                    train_seed=self.train_seed, cv_seed=self.cv_seed, parameters=self.parameters_n,
                                    return_prob_test=True, era_list=self.era_list_n,
                                    show_importance=self.show_importance, show_accuracy=self.show_accuracy,
                                    save_csv_log=True, csv_idx='prejudge_n')

        prob_test = np.zeros_like(self.id_test, dtype=np.float64)

        for idx_p, prob_p in zip(era_idx_test_p, prob_test_p):
            prob_test[idx_p] = prob_p

        for idx_n, prob_n in zip(era_idx_test_n, prob_test_n):
            prob_test[idx_n] = prob_n

        return prob_test

    def train(self, load_pickle=False, load_pickle_path=None):
        """
            Training the model
        """

        start_time = time.time()

        path_list = [self.pred_path + 'positive/',
                     self.pred_path + 'negative/',
                     self.pred_path + 'pred_era/',
                     self.pred_path + 'pred_era/final_results/',
                     self.pred_path + 'final_results/',
                     self.loss_log_path + 'positive/',
                     self.loss_log_path + 'negative/']
        utils.check_dir(path_list)

        print('======================================================')
        print('Start training...')

        if load_pickle is True:

            # Load era_sign_test
            if load_pickle_path is None:
                era_sign_test = utils.load_pkl_to_np(self.prejudged_data_path + 'era_sign_test.p')
            else:
                era_sign_test = utils.load_pkl_to_np(load_pickle_path)

        else:

            # Training Era Sign
            era_sign_test = self.predict_era_sign()

            # era_sign_test = self.load_era_sign_csv(self.pred_path + 'pred_era/final_results/lgb_result.csv')

            # Save era_sign_test to Pickle File
            utils.save_np_to_pkl(era_sign_test, self.prejudged_data_path + 'era_sign_test.p')

        x_test_p, x_g_test_p, id_test_p, era_idx_test_p, x_test_n, \
            x_g_test_n, id_test_n, era_idx_test_n = self.split_data_by_era_sign(era_sign_test)

        # Training Models by Era Sign
        prob_test = \
            self.train_models_by_era_sign(x_test_p, x_g_test_p, id_test_p, era_idx_test_p,
                                          x_test_n, x_g_test_n, id_test_n, era_idx_test_n)

        # Save Predictions
        utils.save_pred_to_csv(self.pred_path + 'final_results/prejudge_', self.id_test, prob_test)

        total_time = time.time() - start_time
        print('======================================================')
        print('Training Done!')
        print('Total Time: {}s'.format(total_time))
        print('======================================================')


if __name__ == '__main__':

    pass
