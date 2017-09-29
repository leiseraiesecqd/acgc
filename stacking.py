import models
import utils
import time
import numpy as np

class DeepStack:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, x_tr_g, x_te_g,
                 pred_path, stack_output_path, hyper_params, params_l1=None, params_l2=None, params_l3=None):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_g_train = x_tr_g
        self.x_g_test = x_te_g
        self.pred_path = pred_path
        self.stack_output_path = stack_output_path
        self.parameters_l1 = params_l1
        self.parameters_l2 = params_l2
        self.parameters_l3 = params_l3
        self.g_train = x_tr_g[:,-1]
        self.g_test = x_te_g[:, -1]
        self.n_valid = hyper_params['n_valid']
        self.n_era = hyper_params['n_era']
        self.n_epoch = hyper_params['n_epoch']

    def init_models_layer1(self, dnn_l1_params=None):

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

        models_l1 = [
                     LGB_L1,
                     XGB_L1,
                     # AB_L1,
                     # RF_L1,
                     # ET_L1,
                     # GB_L1,
                     DNN_L1
                     ]

        return models_l1

    def init_models_layer2(self, dnn_l2_params=None):

        LGB_L2 = models.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.id_test, self.x_g_train, self.x_g_test)

        # DNN_L2 = models.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
        #                                    self.e_train, self.x_test, self.id_test, dnn_l2_params)

        models_l2 = [
                     LGB_L2,
                     # DNN_L2
                     ]

        return models_l2

    def init_models_layer3(self, dnn_l3_params=None):

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
            print('Training on model:{}/{}'.format(iter_model+1, n_model))

            # Training on each model in models_l1 using one cross validation set
            prob_valid, prob_test, losses = \
                model.stack_train(x_train, y_train, w_train, x_g_train,
                                  x_valid, y_valid, w_valid, x_g_valid, x_test, x_g_test, params[iter_model])

            all_model_valid_prob.append(prob_valid)
            all_model_test_prob.append(prob_test)
            all_model_losses.append(losses)

        # Blenders of one cross validation set
        blender_valid_cv = np.array(all_model_valid_prob, dtype=np.float64)  # (n_model+1) * n_valid_sample
        blender_test_cv = np.array(all_model_test_prob, dtype=np.float64)    # n_model * n_test_sample
        blender_losses_cv = np.array(all_model_losses, dtype=np.float64)     # n_model * n_loss

        return blender_valid_cv, blender_test_cv, blender_losses_cv

    def stacker(self, models_initializer, params, x_train_inputs, y_train_inputs, w_train_inputs,
                e_train_inputs, x_g_train_inputs, x_test, x_g_test,
                cv, n_valid=4, n_era=20, n_epoch=1, dnn_param=None, x_train_reuse=None):

        if n_era%n_valid != 0:
            assert ValueError('n_era must be an integer multiple of n_valid!')

        # Stack Reused Features
        if x_train_reuse is not None:
            print('------------------------------------------------------')
            print('Stacking Reused Features...')
            x_train_inputs = np.concatenate((x_train_inputs, x_train_reuse), axis=1)  # n_sample * (n_feature + n_reuse)
            x_train_group = x_g_train_inputs[:-1]
            x_g_train_inputs = np.column_stack((x_train_inputs, x_train_group))       # n_sample * (n_feature + n_reuse + 1)

        # TODO: Print Shape
        print('======================================================')
        print('x_train_inputs shape:{}'.format(x_train_inputs.shape))
        print('x_test shape:{}'.format(x_test.shape))
        print('======================================================')

        n_cv = int(n_era // n_valid)
        blender_x_outputs = np.array([])
        blender_test_outputs = np.array([])

        for epoch in range(n_epoch):

            print('======================================================')
            print('Training on Epoch: {}/{}'.format(epoch+1, n_epoch))

            # Init models blender
            if dnn_param is not None:
                models_blender = models_initializer(dnn_param)
            else:
                models_blender = models_initializer()
            n_model = len(models_blender)

            counter_cv = 0
            epoch_start_time = time.time()
            blender_valid = np.array([])
            blender_test = np.array([])
            # blender_losses = np.array([])

            for x_train, y_train, w_train, x_g_train, \
                x_valid, y_valid, w_valid, x_g_valid, valid_index in cv.era_k_fold_for_stack(x=x_train_inputs,
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
                blender_test_cv = blender_test_cv.reshape(n_model, 1, -1)  # n_model * 1 * n_test_sample
                if counter_cv == 1:
                    blender_valid = blender_valid_cv
                    blender_test = blender_test_cv
                    # blender_losses = blender_losses_cv
                else:
                    # (n_model + 1) * n_sample
                    blender_valid = np.concatenate((blender_valid, blender_valid_cv), axis=1)
                    # n_model * n_cv * n_test_sample
                    blender_test = np.concatenate((blender_test, blender_test_cv), axis=1)
                    # blender_losses = np.concatenate((blender_losses, blender_losses_cv), axis=1)

            # TODO: Print Shape
            print('======================================================')
            print('blender_valid shape:{}'.format(blender_valid.shape))
            print('blender_test shape:{}'.format(blender_test.shape))
            print('======================================================')

            # Sort blender_valid by valid_index
            print('------------------------------------------------------')
            print('Sorting Validation Blenders...')
            blender_valid_sorted = np.zeros_like(blender_valid, dtype=np.float64)  # n_model*n_x_sample
            for column, idx in enumerate(blender_valid[0]):
                blender_valid_sorted[:, int(idx)] = blender_valid[:, column]
            blender_valid_sorted = np.delete(blender_valid_sorted, 0, axis=0)      # n_model*n_x_sample

            # Calculate average of test_prob
            print('------------------------------------------------------')
            print('Calculating Average of Probabilities of Test Set...')
            blender_test_mean = np.mean(blender_test, axis=1)   # n_model * n_test_sample

            # Transpose blenders
            blender_x_e = blender_valid_sorted.transpose()      # n_sample * n_model
            blender_test_e = blender_test_mean.transpose()      # n_test_sample * n_model

            # TODO: Print Shape
            print('======================================================')
            print('blender_x_e shape:{}'.format(blender_x_e.shape))
            print('blender_test_e shape:{}'.format(blender_test_e.shape))
            print('======================================================')

            if epoch == 0:
                blender_x_outputs = blender_x_e
                blender_test_outputs = blender_test_e
            else:
                # n_sample * (n_model x n_epoch)
                blender_x_outputs = np.concatenate((blender_x_outputs, blender_x_e), axis=1)
                # n_test_sample * (n_model x n_epoch)
                blender_test_outputs = np.concatenate((blender_test_outputs, blender_test_e), axis=1)

            epoch_time = time.time() - epoch_start_time
            print('------------------------------------------------------')
            print('Epoch Done!')
            print('Epoch Time: {}s'.format(epoch_time))

        # Stack Group Features
        print('------------------------------------------------------')
        print('Stacking Group Features...')
        blender_x_g_outputs = np.column_stack((blender_x_outputs, self.g_train))
        blender_test_g_outputs = np.column_stack((blender_test_outputs, self.g_test))

        # TODO: Print Shape
        print('======================================================')
        print('blender_x_outputs:{}'.format(blender_x_outputs.shape))
        print('blender_test_outputs:{}'.format(blender_test_outputs.shape))
        print('blender_x_g_outputs:{}'.format(blender_x_g_outputs.shape))
        print('blender_test_g_outputs:{}'.format(blender_test_g_outputs.shape))
        print('======================================================')

        return blender_x_outputs, blender_test_outputs, blender_x_g_outputs, blender_test_g_outputs

    def save_predict(self, pred_path, test_outputs):

        test_prob = np.mean(test_outputs, axis=1)

        utils.save_pred_to_csv(pred_path, self.id_test, test_prob)

    def stack(self):

        start_time = time.time()

        cv_stack = models.CrossValidation()

        dnn_l1_params = self.parameters_l1[-1]
        # dnn_l2_params = self.parameters_l2[-1]
        # dnn_l3_params = self.parameters_l3[-1]

        # Layer 1
        print('------------------------------------------------------')
        print('Start training layer 1...')
        models_initializer_l1 = self.init_models_layer1

        x_outputs_l1, test_outputs_l1, x_g_outputs_l1, test_g_outputs_l1 \
            = self.stacker(models_initializer_l1, self.parameters_l1, self.x_train, self.y_train,
                           self.w_train, self.e_train, self.x_g_train, self.x_test, self.x_g_test,
                           cv_stack, n_valid=self.n_valid[0], n_era=self.n_era[0],
                           n_epoch=self.n_epoch[0], x_train_reuse=None, dnn_param=dnn_l1_params)

        # Save predicted test prob
        self.save_predict(self.pred_path + 'stack_l1_', test_outputs_l1)

        # Save layer outputs
        utils.save_stack_outputs(self.stack_output_path + 'l1_',
                                 x_outputs_l1, test_outputs_l1, x_g_outputs_l1, test_g_outputs_l1)

        layer1_time = time.time() - start_time
        print('------------------------------------------------------')
        print('Layer 1 Training Done!')
        print('Layer Time: {}s'.format(layer1_time))
        print('======================================================')

        # Layer 2
        print('Start training layer 2...')
        models_initializer_l2 = self.init_models_layer2

        x_reuse = self.x_train[:, :88]

        x_outputs_l2, test_outputs_l2, x_g_outputs_l2, test_g_outputs_l2 \
            = self.stacker(models_initializer_l2, self.parameters_l2, x_outputs_l1, self.y_train,
                           self.w_train, self.e_train, x_g_outputs_l1, test_outputs_l1, test_g_outputs_l1,
                           cv_stack, n_valid=self.n_valid[1], n_era=self.n_era[1],
                           n_epoch=self.n_epoch[1], x_train_reuse=x_reuse, dnn_param=None)

        # Save predicted test prob
        self.save_predict(self.pred_path + 'stack_l2_', test_outputs_l2)

        # Save layer outputs
        utils.save_stack_outputs(self.stack_output_path + 'l2_',
                                 x_outputs_l2, test_outputs_l2, x_g_outputs_l2, test_g_outputs_l2)

        layer2_time = time.time() - start_time
        print('------------------------------------------------------')
        print('Layer 2 Training Done!')
        print('Layer Time: {}s'.format(layer2_time))
        print('======================================================')

        # # Layer 3
        # print('Start training layer 3...')
        # models_initializer_l3 = self.init_models_layer3
        #
        # x_outputs_l3, test_outputs_l3, x_g_outputs_l3, test_g_outputs_l3 \
        #     = self.stacker(models_initializer_l3, self.parameters_l3, x_outputs_l2, self.y_train,
        #                    self.w_train, self.e_train, x_g_outputs_l2, test_outputs_l2, test_g_outputs_l2,
        #                    cv_stack, n_valid=self.n_valid[2], n_era=self.n_era[2],
        #                    n_epoch=self.n_epoch[2], x_train_reuse=None, dnn_param=None)
        #
        # # Save predicted test prob
        # self.save_predict(self.pred_path + 'stack_l3_', test_outputs_l3)
        #
        # # Save layer outputs
        # utils.save_stack_outputs(self.stack_output_path + 'l3_',
        #                          x_outputs_l3, test_outputs_l3, x_g_outputs_l3, test_g_outputs_l3)
        #
        # layer3_time = time.time() - start_time
        # print('------------------------------------------------------')
        # print('Layer3 Training Done!')
        # print('Layer Time: {}s'.format(layer3_time))
        # print('======================================================')

        # Save predicted test prob
        final_result = test_outputs_l2
        self.save_predict(self.pred_path + 'final_results/stack_', final_result)

        total_time = time.time() - start_time
        print('======================================================')
        print('Training Done!')
        print('Total Time: {}s'.format(total_time))
        print('======================================================')


class TreeStack:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, x_tr_g, x_te_g,
                 pred_path, stack_output_path, hyper_params, params_l1, params_l2, params_l3):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_g_train = x_tr_g
        self.x_g_test = x_te_g
        self.pred_path = pred_path
        self.stack_output_path = stack_output_path
        self.parameters_l1 = params_l1
        self.parameters_l2 = params_l2
        self.parameters_l3 = params_l3
        self.g_train = x_tr_g[:, -1]
        self.g_test = x_te_g[:, -1]
        self.n_valid = hyper_params['n_valid']
        self.n_era = hyper_params['n_era']
        self.n_epoch = hyper_params['n_epoch']

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

        models_l1 = [LGB_L1,
                     # XGB_L1,
                     # AB_L1,
                     # RF_L1,
                     # ET_L1,
                     # GB_L1,
                     DNN_L1
                     ]

        return models_l1

    def init_models_layer2(self, dnn_l2_params):

        LGB_L2 = models.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.id_test, self.x_g_train, self.x_g_test)

        DNN_L2 = models.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
                                           self.e_train, self.x_test, self.id_test, dnn_l2_params)

        models_l2 = [LGB_L2,
                     DNN_L2]

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
            print('Training on model:{}/{}'.format(iter_model + 1, n_model))

            # Training on each model in models_l1 using one cross validation set
            prob_valid, prob_test, losses = \
                model.stack_train(x_train, y_train, w_train, x_g_train,
                                  x_valid, y_valid, w_valid, x_g_valid, x_test, x_g_test, params[iter_model])

            all_model_valid_prob.append(prob_valid)
            all_model_test_prob.append(prob_test)
            all_model_losses.append(losses)

        # Blenders of one cross validation set
        blender_valid_cv = np.array(all_model_valid_prob, dtype=np.float64)  # (n_model+1) * n_valid_sample
        blender_test_cv = np.array(all_model_test_prob, dtype=np.float64)  # n_model * n_test_sample
        blender_losses_cv = np.array(all_model_losses, dtype=np.float64)  # n_model * n_loss

        return blender_valid_cv, blender_test_cv, blender_losses_cv

    def stacker(self, models_initializer, dnn_param, params, x_train_inputs, y_train_inputs, w_train_inputs,
                e_train_inputs, x_g_train_inputs, x_test, x_g_test,
                cv, n_valid=4, n_era=20, n_epoch=1, x_train_reuse=None):

        if n_era % n_valid != 0:
            assert ValueError('n_era must be an integer multiple of n_valid!')

        # Stack Reused Features
        if x_train_reuse is not None:
            print('------------------------------------------------------')
            print('Stacking Reused Features...')
            x_train_inputs = np.concatenate((x_train_inputs, x_train_reuse), axis=1)  # n_sample * (n_feature + n_reuse)
            x_train_group = x_g_train_inputs[:-1]
            x_g_train_inputs = np.column_stack((x_train_inputs, x_train_group))  # n_sample * (n_feature + n_reuse + 1)

        # TODO: Print Shape
        print('======================================================')
        print('x_train_inputs shape:{}'.format(x_train_inputs.shape))
        print('x_test shape:{}'.format(x_test.shape))
        print('======================================================')

        n_cv = int(n_era // n_valid)
        blender_x_outputs = np.array([])
        blender_test_outputs = np.array([])

        for epoch in range(n_epoch):

            print('======================================================')
            print('Training on Epoch: {}/{}'.format(epoch + 1, n_epoch))

            # Init models blender
            if dnn_param is not None:
                models_blender = models_initializer(dnn_param)
            else:
                models_blender = models_initializer()
            n_model = len(models_blender)

            counter_cv = 0
            epoch_start_time = time.time()
            blender_valid = np.array([])
            blender_test = np.array([])
            # blender_losses = np.array([])

            for x_train, y_train, w_train, x_g_train, \
                x_valid, y_valid, w_valid, x_g_valid, valid_index in cv.era_k_fold_for_stack(x=x_train_inputs,
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
                blender_test_cv = blender_test_cv.reshape(n_model, 1, -1)  # n_model * 1 * n_test_sample
                if counter_cv == 1:
                    blender_valid = blender_valid_cv
                    blender_test = blender_test_cv
                    # blender_losses = blender_losses_cv
                else:
                    # (n_model + 1) * n_sample
                    blender_valid = np.concatenate((blender_valid, blender_valid_cv), axis=1)
                    # n_model * n_cv * n_test_sample
                    blender_test = np.concatenate((blender_test, blender_test_cv), axis=1)
                    # blender_losses = np.concatenate((blender_losses, blender_losses_cv), axis=1)

            # TODO: Print Shape
            print('======================================================')
            print('blender_valid shape:{}'.format(blender_valid.shape))
            print('blender_test shape:{}'.format(blender_test.shape))
            print('======================================================')

            # Sort blender_valid by valid_index
            print('------------------------------------------------------')
            print('Sorting Validation Blenders...')
            blender_valid_sorted = np.zeros_like(blender_valid, dtype=np.float64)  # n_model*n_x_sample
            for column, idx in enumerate(blender_valid[0]):
                blender_valid_sorted[:, int(idx)] = blender_valid[:, column]
            blender_valid_sorted = np.delete(blender_valid_sorted, 0, axis=0)  # n_model*n_x_sample

            # Calculate average of test_prob
            print('------------------------------------------------------')
            print('Calculating Average of Probabilities of Test Set...')
            blender_test_mean = np.mean(blender_test, axis=1)  # n_model * n_test_sample

            # Transpose blenders
            blender_x_e = blender_valid_sorted.transpose()  # n_sample * n_model
            blender_test_e = blender_test_mean.transpose()  # n_test_sample * n_model

            # TODO: Print Shape
            print('======================================================')
            print('blender_x_e shape:{}'.format(blender_x_e.shape))
            print('blender_test_e shape:{}'.format(blender_test_e.shape))
            print('======================================================')

            if epoch == 0:
                blender_x_outputs = blender_x_e
                blender_test_outputs = blender_test_e
            else:
                # n_sample * (n_model x n_epoch)
                blender_x_outputs = np.concatenate((blender_x_outputs, blender_x_e), axis=1)
                # n_test_sample * (n_model x n_epoch)
                blender_test_outputs = np.concatenate((blender_test_outputs, blender_test_e), axis=1)

            epoch_time = time.time() - epoch_start_time
            print('------------------------------------------------------')
            print('Epoch Done!')
            print('Epoch Time: {}s'.format(epoch_time))

        # Stack Group Features
        print('------------------------------------------------------')
        print('Stacking Group Features...')
        blender_x_g_outputs = np.column_stack((blender_x_outputs, self.g_train))
        blender_test_g_outputs = np.column_stack((blender_test_outputs, self.g_test))

        # TODO: Print Shape
        print('======================================================')
        print('blender_x_outputs:{}'.format(blender_x_outputs.shape))
        print('blender_test_outputs:{}'.format(blender_test_outputs.shape))
        print('blender_x_g_outputs:{}'.format(blender_x_g_outputs.shape))
        print('blender_test_g_outputs:{}'.format(blender_test_g_outputs.shape))
        print('======================================================')

        return blender_x_outputs, blender_test_outputs, blender_x_g_outputs, blender_test_g_outputs

    def iterator(self):

        pass

    def save_predict(self, pred_path, test_outputs):

        test_prob = np.mean(test_outputs, axis=1)

        utils.save_pred_to_csv(pred_path, self.id_test, test_prob)

    def stack(self):

        start_time = time.time()

        cv_stack = models.CrossValidation()

        dnn_l1_params = self.parameters_l1[-1]
        dnn_l2_params = self.parameters_l2[-1]
        dnn_l3_params = self.parameters_l3[-1]

        # Layer 1
        print('------------------------------------------------------')
        print('Start training layer 1...')
        models_initializer_l1 = self.init_models_layer1

        x_outputs_l1, test_outputs_l1, x_g_outputs_l1, test_g_outputs_l1 \
            = self.stacker(models_initializer_l1, dnn_l1_params, self.parameters_l1, self.x_train, self.y_train,
                           self.w_train, self.e_train, self.x_g_train, self.x_test, self.x_g_test,
                           cv_stack, n_valid=self.n_valid[0], n_era=self.n_era[0],
                           n_epoch=self.n_epoch[0], x_train_reuse=None)

        # Save predicted test prob
        self.save_predict(self.pred_path + 'stack_l1_', test_outputs_l1)

        # Save layer outputs
        utils.save_stack_outputs(self.stack_output_path + 'l1_',
                                 x_outputs_l1, test_outputs_l1, x_g_outputs_l1, test_g_outputs_l1)

        layer1_time = time.time() - start_time
        print('------------------------------------------------------')
        print('Layer 1 Training Done!')
        print('Layer Time: {}s'.format(layer1_time))
        print('======================================================')

        # Layer 2
        print('Start training layer 2...')
        models_initializer_l2 = self.init_models_layer2

        x_outputs_l2, test_outputs_l2, x_g_outputs_l2, test_g_outputs_l2 \
            = self.stacker(models_initializer_l2, dnn_l2_params, self.parameters_l2, x_outputs_l1, self.y_train,
                           self.w_train, self.e_train, x_g_outputs_l1, test_outputs_l1, test_g_outputs_l1,
                           cv_stack, n_valid=self.n_valid[1], n_era=self.n_era[1],
                           n_epoch=self.n_epoch[1], x_train_reuse=None)

        # Save predicted test prob
        self.save_predict(self.pred_path + 'stack_l2_', test_outputs_l2)

        # Save layer outputs
        utils.save_stack_outputs(self.stack_output_path + 'l2_',
                                 x_outputs_l2, test_outputs_l2, x_g_outputs_l2, test_g_outputs_l2)

        layer2_time = time.time() - start_time
        print('------------------------------------------------------')
        print('Layer 2 Training Done!')
        print('Layer Time: {}s'.format(layer2_time))
        print('======================================================')

        # Layer 3
        print('Start training layer 3...')
        models_initializer_l3 = self.init_models_layer3

        x_outputs_l3, test_outputs_l3, x_g_outputs_l3, test_g_outputs_l3 \
            = self.stacker(models_initializer_l3, dnn_l3_params, self.parameters_l3, x_outputs_l2, self.y_train,
                           self.w_train, self.e_train, x_g_outputs_l2, test_outputs_l2, test_g_outputs_l2,
                           cv_stack, n_valid=self.n_valid[2], n_era=self.n_era[2],
                           n_epoch=self.n_epoch[2], x_train_reuse=None)

        # Save predicted test prob
        self.save_predict(self.pred_path + 'stack_l3_', test_outputs_l3)

        # Save layer outputs
        utils.save_stack_outputs(self.stack_output_path + 'l3_',
                                 x_outputs_l3, test_outputs_l3, x_g_outputs_l3, test_g_outputs_l3)

        layer3_time = time.time() - start_time
        print('------------------------------------------------------')
        print('Layer3 Training Done!')
        print('Layer Time: {}s'.format(layer3_time))
        print('======================================================')

        # Save predicted test prob
        final_result = test_outputs_l3
        self.save_predict(self.pred_path + 'final_results/stack_', final_result)

        total_time = time.time() - start_time
        print('======================================================')
        print('Training Done!')
        print('Total Time: {}s'.format(total_time))
        print('======================================================')


if __name__ == '__main__':

    pass
