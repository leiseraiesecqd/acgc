import model
import numpy as np
from model import CrossValidation


class Stacking:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, x_tr_g, x_te_g, pred_path, params_l1, params_l2):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_train_g = x_tr_g
        self.x_test_g = x_te_g
        self.pred_path = pred_path
        self.parameters_l1 = params_l1
        self.parameters_l2 = params_l2

    def init_models_layer1(self, dnn_l1_params):

        LGB_L1 = model.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                                self.x_test, self.id_test, self.x_train_g, self.x_test_g)
        XGB_L1 = model.XGBoost(self.x_train, self.y_train, self.w_train,
                               self.e_train, self.x_test, self.id_test)
        AB_L1 = model.AdaBoost(self.x_train, self.y_train, self.w_train,
                               self.e_train, self.x_test, self.id_test)
        RF_L1 = model.RandomForest(self.x_train, self.y_train, self.w_train,
                                   self.e_train, self.x_test, self.id_test)
        ET_L1 = model.ExtraTrees(self.x_train, self.y_train, self.w_train,
                                 self.e_train, self.x_test, self.id_test)
        GB_L1 = model.GradientBoosting(self.x_train, self.y_train, self.w_train,
                                       self.e_train, self.x_test, self.id_test)
        DNN_L1 = model.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
                                          self.e_train, self.x_test, self.id_test, dnn_l1_params)

        models_l1 = [LGB_L1, XGB_L1, AB_L1, RF_L1, ET_L1, GB_L1, DNN_L1]

        return models_l1

    def init_models_layer2(self, dnn_l2_params, x_train_g_l2, x_test_g_l2):

        DNN_L2 = model.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
                                          self.e_train, self.x_test, self.id_test, dnn_l2_params)

        LGB_L2 = model.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                                self.x_test, self.id_test, x_train_g_l2, x_test_g_l2)

        models_l2 = [DNN_L2, LGB_L2]

        return models_l2

    def train_models(self, models, params, x_train, y_train, w_train, x_g_train,
                    x_valid, y_valid, w_valid, x_g_valid, idx_valid):

        # First raw - idx_valid
        all_model_valid_prob = [idx_valid]
        all_model_test_prob = []
        all_model_losses = []

        for iter_model, model in enumerate(models):

            # Training on each model in models_l1 using one cross validation set
            prob_valid, prob_test, losses = \
                model.stastack_train(x_train, y_train, w_train, x_g_train,
                                     x_valid, y_valid, w_valid, x_g_valid, params[iter_model])

            all_model_valid_prob.append(prob_valid)
            all_model_test_prob.append(prob_test)
            all_model_losses.append(losses)

        # Blenders of one cross validation set
        blender_valid_layer1_cv = np.array(all_model_valid_prob, dtype=np.float64)
        blender_test_layer1_cv = np.array(all_model_test_prob, dtype=np.float64)
        blender_losses_layer1_cv = np.array(all_model_losses, dtype=np.float64)

        return blender_valid_layer1_cv, blender_test_layer1_cv, blender_losses_layer1_cv

    def train_layer(self, models, params, CV_stack, n_valid, n_cv_l1):

        counter_layer1_cv = 0

        blender_valid_layer1 = np.array([])
        blender_test_layer1 = np.array([])
        blender_losses_layer1 = np.array([])

        for x_train, y_train, w_train, x_g_train, \
            x_valid, y_valid, w_valid, x_g_valid, valid_index in CV_stack.era_k_fold_for_stack(x=self.x_train,
                                                                                               y=self.y_train,
                                                                                               w=self.w_train,
                                                                                               e=self.e_train,
                                                                                               x_g=self.x_train_g,
                                                                                               n_valid=n_valid,
                                                                                               n_cv=n_cv_l1):
            counter_layer1_cv += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(counter_layer1_cv, n_cv_l1))

            # Training on models and get blenders of one cross validation set
            blender_valid_layer1_cv, blender_test_layer1_cv, \
            blender_losses_layer1_cv = self.train_models(models, params, x_train, y_train, w_train, x_g_train,
                                                         x_valid, y_valid, w_valid, x_g_valid, valid_index)

            # Add blenders of one cross validation set to blenders of all CV
            if counter_layer1_cv == 1:
                blender_valid_layer1 = blender_valid_layer1_cv
                blender_test_layer1 = blender_test_layer1_cv
                blender_losses_layer1 = blender_losses_layer1_cv
            else:
                blender_valid_layer1 = np.concatenate((blender_valid_layer1, blender_valid_layer1_cv), axis=1)
                blender_test_layer1 = np.concatenate((blender_test_layer1, blender_test_layer1_cv), axis=1)
                blender_losses_layer1 = np.concatenate((blender_losses_layer1, blender_losses_layer1_cv), axis=1)

        # Sort blender_valid by valid_index
        blender_valid_layer1_sorted = np.zeros_like(blender_valid_layer1, dtype=np.float64)
        for column, idx in enumerate(blender_valid_layer1[0]):
            blender_valid_layer1_sorted[:, idx] = blender_valid_layer1[:, column]
        blender_valid_layer1_sorted = np.delete(blender_valid_layer1_sorted, 0, axis=0)

        x_train_l2 = blender_valid_layer1_sorted.transpose()
        x_test_l2 = blender_test_layer1.transpose()

        g_train = self.x_train_g[:, -1]
        g_test = self.x_test_g[:, -1]
        x_train_g_l2 = np.column_stack((x_train_l2, g_train))
        x_test_g_l2 = np.column_stack((x_test_l2, g_test))

        return x_train_l2, x_test_l2, x_train_g_l2, x_test_g_l2

    def stack(self, pred_path=None, n_valid=4, n_cv_l1=5, n_cv_l2=10, n_valid_l2=4, n_cv_l2=16):

        CV_stack = model.CrossValidation()

        dnn_l1_params = self.parameters_l1[-1]
        dnn_l2_params = self.parameters_l2[-1]

        counter_layer2 = 0

        for iter_l2 in range(n_cv_l2):

            counter_layer2 += 1

            models_l1 = self.init_models_layer1(dnn_l1_params)

            x_train_l2, x_test_l2, \
                x_train_g_l2, x_test_g_l2 = self.train_layer(models_l1, self.parameters_l1,
                                                             CV_stack, n_valid, n_cv_l1)

            models_l2 = self.init_models_layer2(dnn_l2_params, x_train_g_l2, x_test_g_l2)

            x_train_l3, x_test_l3, \
                x_train_g_l3, x_test_g_l3 = self.train_layer(models_l1, self.parameters_l1,
                                                             CV_stack, n_valid, n_cv_l2)














