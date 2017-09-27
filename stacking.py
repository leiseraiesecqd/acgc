import model
import numpy
from model import CrossValidation


class Layer1:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, x_tr_g, x_te_g, params):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_train_g = x_tr_g
        self.x_test_g = x_te_g
        self.parameters = params
        self.models = []

        self.init_models()

    def init_models(self):

        LGB = model.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                             self.x_test, self.id_test, self.x_train_g, self.x_test_g)
        XGB = model.XGBoost(self.x_train, self.y_train, self.w_train,
                            self.e_train, self.x_test, self.id_test)
        AB = model.AdaBoost(self.x_train, self.y_train, self.w_train,
                            self.e_train, self.x_test, self.id_test)
        RF = model.RandomForest(self.x_train, self.y_train, self.w_train,
                                self.e_train, self.x_test, self.id_test)
        ET = model.ExtraTrees(self.x_train, self.y_train, self.w_train,
                              self.e_train, self.x_test, self.id_test)
        GB = model.GradientBoosting(self.x_train, self.y_train, self.w_train,
                                    self.e_train, self.x_test, self.id_test)

        self.models = [LGB, XGB, AB, RF, ET, GB]

    def layer1(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid, w_valid, x_g_valid):

        for iter_model, model in enumerate(self.models):

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))

            prob_valid_clf, prob_test_clf, loss_clf = \
                model.stastack_train(self.count, x_train, y_train, w_train, x_g_train, x_valid,
                                     y_valid, w_valid, x_g_valid, self.pred_path, self.parameters[iter_model])




    def train_by_cv(self, n_valid, n_cv_l1, n_cv_l2, pred_path):

        n_cv = n_cv_l1 * n_cv_l2

        models = self.init_models()

        count = 0

        for x_train, y_train, w_train, x_g_train, \
            x_valid, y_valid, w_valid, x_g_valid in CrossValidation.era_k_fold_with_category(x=self.x_train,
                                                                                             y=self.y_train,
                                                                                             w=self.w_train,
                                                                                             e=self.e_train,
                                                                                             x_g=self.x_train_g,
                                                                                             n_valid=n_valid,
                                                                                             n_cv=n_cv):
            count += 1





