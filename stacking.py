import models
import utils
import time
import numpy as np

class Stacking:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, x_tr_g, x_te_g,
                 pred_path, params_l1, params_l2, params_l3):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_g_train = x_tr_g
        self.x_g_test = x_te_g
        self.pred_path = pred_path
        self.parameters_l1 = params_l1
        self.parameters_l2 = params_l2
        self.parameters_l3 = params_l3
        self.g_train = x_tr_g[:,-1]
        self.g_test = x_te_g[:, -1]

    def init_models_layer1(self, dnn_l1_params):

        LGB_L1 = models.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.id_test, self.x_g_train, self.x_g_test)
        XGB_L1 = models.XGBoost(self.x_train, self.y_train, self.w_train,
                                self.e_train, self.x_test, self.id_test)
        AB_L1 = models.AdaBoost(self.x_train, self.y_train, self.w_train,
                                self.e_train, self.x_test, self.id_test)
        RF_L1 = models.RandomForest(self.x_train, self.y_train, self.w_train,
                                    self.e_train, self.x_test, self.id_test)
        ET_L1 = models.ExtraTrees(self.x_train, self.y_train, self.w_train,
                                  self.e_train, self.x_test, self.id_test)
        GB_L1 = models.GradientBoosting(self.x_train, self.y_train, self.w_train,
                                        self.e_train, self.x_test, self.id_test)
        DNN_L1 = models.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
                                           self.e_train, self.x_test, self.id_test, dnn_l1_params)

        models_l1 = [LGB_L1, XGB_L1, AB_L1, RF_L1, ET_L1, GB_L1, DNN_L1]

        return models_l1

    def init_models_layer2(self, dnn_l2_params):

        LGB_L2 = models.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.id_test, self.x_g_train, self.x_g_test)

        DNN_L2 = models.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
                                           self.e_train, self.x_test, self.id_test, dnn_l2_params)

        models_l2 = [LGB_L2, DNN_L2]

        return models_l2

    def init_models_layer3(self, dnn_l3_params):

        DNN_L3 = models.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
                                           self.e_train, self.x_test, self.id_test, dnn_l3_params)

        models_l3 = [DNN_L3]

        return models_l3

    def train_models(self, models_blender, params, x_train, y_train, w_train, x_g_train,
                     x_valid, y_valid, w_valid, x_g_valid, idx_valid, x_test, x_g_test):

        # First raw - idx_valid
        all_model_valid_prob = [idx_valid]
        all_model_test_prob = []
        all_model_losses = []

        n_model = len(models_blender)

        for iter_model, model in enumerate(models_blender):

            print('------------------------------------------------------')
            print('Training on model:{}/{}'.format(iter_model, n_model))

            # Training on each model in models_l1 using one cross validation set
            prob_valid, prob_test, losses = \
                model.stastack_train(x_train, y_train, w_train, x_g_train,
                                     x_valid, y_valid, w_valid, x_g_valid, x_test, x_g_test, params[iter_model])

            all_model_valid_prob.append(prob_valid)
            all_model_test_prob.append(prob_test)
            all_model_losses.append(losses)

        # Blenders of one cross validation set
        blender_valid_cv = np.array(all_model_valid_prob, dtype=np.float64)  # (n_model+1) * n_valid_sample
        blender_test_cv = np.array(all_model_test_prob, dtype=np.float64)    # n_model * n_test_sample
        blender_losses_cv = np.array(all_model_losses, dtype=np.float64)     # n_model * n_loss

        return blender_valid_cv, blender_test_cv, blender_losses_cv

    def train_layer(self, models_blender, params, x_train_inputs, y_train_inputs, w_train_inputs,
                    e_train_inputs, x_g_train_inputs, x_test, x_g_test,
                    CV, n_valid=4, n_era=20, n_epoch=1, x_train_reuse=None):

        if n_era%n_valid != 0:
            assert ValueError('n_era must be an integer multiple of n_valid!')

        # Stack Reused Features
        print('------------------------------------------------------')
        print('Stacking Reused Features...')
        x_train_inputs = np.column_stack((x_train_inputs, x_train_reuse))
        x_g_train_inputs = np.column_stack((x_g_train_inputs, x_train_reuse))

        n_cv = int(n_era // n_valid)
        blender_x_prob = np.array([])
        blender_test_prob = np.array([])

        for epoch in range(n_epoch):

            print('======================================================')
            print('Training on Epoch: {}/{}'.format(epoch+1, n_epoch))

            counter_cv = 0
            blender_valid = np.array([])
            blender_test = np.array([])
            # blender_losses = np.array([])

            for x_train, y_train, w_train, x_g_train, \
                x_valid, y_valid, w_valid, x_g_valid, valid_index in CV.era_k_fold_for_stack(x=x_train_inputs,
                                                                                             y=y_train_inputs,
                                                                                             w=w_train_inputs,
                                                                                             e=e_train_inputs,
                                                                                             x_g=x_g_train_inputs,
                                                                                             n_valid=n_valid,
                                                                                             n_cv=n_cv):
                counter_cv += 1

                print('======================================================')
                print('Training on Cross Validation Set: {}/{}'.format(counter_cv, n_cv))

                # Training on models and get blenders of one cross validation set
                blender_valid_cv, blender_test_cv, \
                    blender_losses_cv = self.train_models(models_blender, params, x_train, y_train, w_train,
                                                          x_g_train, x_valid, y_valid, w_valid, x_g_valid,
                                                          valid_index, x_test, x_g_test)

                # Add blenders of one cross validation set to blenders of all CV
                if counter_cv == 1:
                    blender_valid = blender_valid_cv
                    blender_test = blender_test_cv
                    # blender_losses = blender_losses_cv
                else:
                    blender_valid = np.concatenate((blender_valid, blender_valid_cv), axis=1)  # (n_model + 1) * n_sample
                    blender_test = np.concatenate((blender_test, blender_test_cv), axis=0)     # (n_model x n_cv) * n_test_sample
                    # blender_losses = np.concatenate((blender_losses, blender_losses_cv), axis=1)

            # Sort blender_valid by valid_index
            print('------------------------------------------------------')
            print('Sorting Validation Blenders...')
            blender_valid_sorted = np.zeros_like(blender_valid, dtype=np.float64)  # n_model*n_sample
            for column, idx in enumerate(blender_valid[0]):
                blender_valid_sorted[:, idx] = blender_valid[:, column]
            blender_valid_sorted = np.delete(blender_valid_sorted, 0, axis=0)  # n_model*n_sample

            # Transpose blenders
            blender_x_e = blender_valid_sorted.transpose()  # n_sample * n_model
            blender_test_e = blender_test.transpose()           # n_test_sample * (n_model x n_cv)

            if epoch == 0:
                blender_x_prob = blender_x_e
                blender_test_prob = blender_test_e
            else:
                blender_x_prob = np.concatenate((blender_x_prob, blender_x_e), axis=1) # n_sample * (n_model x n_epoch)
                blender_test_prob = np.concatenate((blender_test_prob, blender_test_e), axis=1)    # n_test_sample * (n_model x n_cv x n_epoch)

        # Calculate average of test_prob
        print('------------------------------------------------------')
        print('Calculating Average of Probabilities of Test Set...')
        blender_test_outputs = np.mean(blender_test_prob, axis=1)  # vector: n_test_sample
        blender_x_outputs = blender_x_prob

        # Stack Group Features
        print('------------------------------------------------------')
        print('Stacking Group Features...')
        blender_x_g_outputs = np.column_stack((blender_x_outputs, self.g_train))
        blender_test_g_outputs = np.column_stack((blender_test_outputs, self.g_test))

        return blender_x_outputs, blender_test_outputs, blender_x_g_outputs, blender_test_g_outputs

    def stack(self):

        start_time = time.time()

        CV_stack = models.CrossValidation()

        dnn_l1_params = self.parameters_l1[-1]
        dnn_l2_params = self.parameters_l2[-1]
        dnn_l3_params = self.parameters_l3[-1]

        # Layer 1
        print('----------------------------------------------')
        print('Start training layer 1...')
        models_l1 = self.init_models_layer1(dnn_l1_params)

        x_outputs_l1, test_outputs_l1, x_g_outputs_l1, test_g_outputs_l1 \
            = self.train_layer(models_l1, self.parameters_l1, self.x_train, self.y_train, self.w_train,
                               self.e_train, self.x_g_train, self.x_test, self.x_g_test,
                               CV_stack, n_valid=4, n_era=20, n_epoch=1, x_train_reuse=None)

        utils.save_pred_to_csv(self.pred_path + 'stack_l1_', self.id_test, test_outputs_l1)

        layer1_time = time.time() - start_time
        print('==============================================')
        print('Layer 1 Training Finished!')
        print('Time: {}s'.format(layer1_time))
        print('==============================================')


        # Layer 2
        print('Start training layer 2...')
        models_l2 = self.init_models_layer2(dnn_l2_params)

        x_outputs_l2, test_outputs_l2, x_g_outputs_l2, test_g_outputs_l2 \
            = self.train_layer(models_l2, self.parameters_l2, x_outputs_l1, self.y_train, self.w_train,
                               self.e_train, x_g_outputs_l1, test_outputs_l1, test_g_outputs_l1,
                               CV_stack, n_valid=4, n_era=20, n_epoch=1, x_train_reuse=None)

        utils.save_pred_to_csv(self.pred_path + 'stack_l2_', self.id_test, test_outputs_l2)

        layer2_time = time.time() - start_time
        print('----------------------------------------------')
        print('Layer 2 Training Finished!')
        print('Time: {}s'.format(layer2_time))
        print('==============================================')

        # Layer 3
        print('Start training layer 3...')
        models_l3 = self.init_models_layer3(dnn_l3_params)

        _, test_outputs_l3, _, _ \
            = self.train_layer(models_l3, self.parameters_l3, x_outputs_l2, self.y_train, self.w_train,
                               self.e_train, x_g_outputs_l2, test_outputs_l2, test_g_outputs_l2,
                               CV_stack, n_valid=4, n_era=20, n_epoch=1, x_train_reuse=None)

        utils.save_pred_to_csv(self.pred_path + 'final_results/stack_', self.id_test, test_outputs_l3)

        layer3_time = time.time() - start_time
        print('----------------------------------------------')
        print('Layer3 Training Finished!')
        print('Time: {}s'.format(layer3_time))
        print('==============================================')

        total_time = time.time() - start_time
        print('==============================================')
        print('Training Finished!')
        print('Toal Time: {}s'.format(total_time))
        print('==============================================')


if __name__ == '__main__':

    pass
