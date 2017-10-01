import models
import utils
import time
import numpy as np


# Deep Stack
class DeepStack:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, x_tr_g, x_te_g,
                 pred_path, loss_log_path, stack_output_path, hyper_params, layers_param):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_g_train = x_tr_g
        self.x_g_test = x_te_g
        self.pred_path = pred_path
        self.loss_log_path = loss_log_path
        self.stack_output_path = stack_output_path
        self.layers_param = layers_param
        self.g_train = x_tr_g[:,-1]
        self.g_test = x_te_g[:, -1]
        self.n_valid = hyper_params['n_valid']
        self.n_era = hyper_params['n_era']
        self.n_epoch = hyper_params['n_epoch']
        self.cv_seed = hyper_params['cv_seed']

    def init_models_layer1(self, dnn_l1_params=None):

        LGB_L1 = models.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.id_test, self.x_g_train, self.x_g_test)
        XGB_L1 = models.XGBoost(self.x_train, self.y_train, self.w_train,
                                self.e_train, self.x_test, self.id_test)
        # AB_L1 = models.AdaBoost(self.x_train, self.y_train, self.w_train,
        #                         self.e_train, self.x_test, self.id_test)
        # RF_L1 = models.RandomForest(self.x_train, self.y_train, self.w_train,
        #                             self.e_train, self.x_test, self.id_test)
        # ET_L1 = models.ExtraTrees(self.x_train, self.y_train, self.w_train,
        #                           self.e_train, self.x_test, self.id_test)
        # GB_L1 = models.GradientBoosting(self.x_train, self.y_train, self.w_train,
        #                                 self.e_train, self.x_test, self.id_test)
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

    @staticmethod
    def train_models(models_blender, params, x_train, y_train, w_train, x_g_train,
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
                cv, n_valid=4, n_era=20, cv_seed=None, n_epoch=1,
                x_train_reuse=None, x_test_reuse=None, dnn_param=None, ):

        if n_era%n_valid != 0:
            assert ValueError('n_era must be an integer multiple of n_valid!')

        # Stack Reused Features
        if x_train_reuse is not None:
            print('------------------------------------------------------')
            print('Stacking Reused Features...')
            x_train_inputs = np.concatenate((x_train_inputs, x_train_reuse), axis=1)  # n_sample * (n_feature + n_reuse)
            x_train_group = x_g_train_inputs[:, -1]
            x_g_train_inputs = np.column_stack((x_train_inputs, x_train_group))       # n_sample * (n_feature + n_reuse + 1)

        if x_test_reuse is not None:
            print('------------------------------------------------------')
            print('Stacking Reused Features...')
            x_test = np.concatenate((x_test, x_test_reuse), axis=1)  # n_sample * (n_feature + n_reuse)
            x_test_group = x_g_test[:, -1]
            x_g_test = np.column_stack((x_test, x_test_group))       # n_sample * (n_feature + n_reuse + 1)

        # Print Shape
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

            for x_train, y_train, w_train, x_g_train, x_valid, y_valid, \
                w_valid, x_g_valid, valid_index, valid_era in cv.era_k_fold_for_stack(x=x_train_inputs,
                                                                                      y=y_train_inputs,
                                                                                      w=w_train_inputs,
                                                                                      e=e_train_inputs,
                                                                                      x_g=x_g_train_inputs,
                                                                                      n_valid=n_valid,
                                                                                      n_cv=n_cv,
                                                                                      seed=cv_seed):
                counter_cv += 1

                print('======================================================')
                print('Training on Cross Validation Set: {}/{}'.format(counter_cv, n_cv))
                print('Validation Set Era: ', valid_era)

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

            # Print Shape
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

            # Print Shape
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

        # Print Shape
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

    def stack_outputs_train(self, model_name, params, n_valid, n_cv, x_outputs, test_outputs, x_g_outputs, test_g_outputs):

        if model_name == 'LGB':
            model = models.LightGBM(x_outputs, self.y_train,  self.w_train,  self.e_train,
                                    test_outputs,  self.id_test, x_g_outputs, test_g_outputs)
        elif model_name == 'DNN':
            model = models.DeepNeuralNetworks(x_outputs, self.y_train,  self.w_train,  self.e_train,
                                              test_outputs,  self.id_test, params)
        else:
            model = 0
            assert ValueError('Wrong model name!')

        print('Start training {}...'.format(model_name))

        model.train_sklearn(self.pred_path + 'stack_outputs/',  self.loss_log_path, n_valid=n_valid, n_cv=n_cv, parameters=params)

    def stack(self):

        start_time = time.time()

        cv_stack = models.CrossValidation()

        dnn_l1_params = self.layers_param[0][-1]
        # dnn_l2_params = self.layers_param[1][-1]
        # dnn_l3_params = self.layers_param[2][-1]

        # Layer 1
        print('------------------------------------------------------')
        print('Start training layer 1...')
        models_initializer_l1 = self.init_models_layer1

        x_outputs_l1, test_outputs_l1, x_g_outputs_l1, test_g_outputs_l1 \
            = self.stacker(models_initializer_l1, self.layers_param[0], self.x_train, self.y_train,
                           self.w_train, self.e_train, self.x_g_train, self.x_test, self.x_g_test,
                           cv_stack, n_valid=self.n_valid[0], n_era=self.n_era[0], cv_seed=self.cv_seed,
                           n_epoch=self.n_epoch[0], dnn_param=dnn_l1_params)

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

        x_tr_reuse = self.x_train[:, :88]
        x_te_reuse = self.x_test[:, :88]

        x_outputs_l2, test_outputs_l2, x_g_outputs_l2, test_g_outputs_l2 \
            = self.stacker(models_initializer_l2, self.layers_param[1], x_outputs_l1, self.y_train,
                           self.w_train, self.e_train, x_g_outputs_l1, test_outputs_l1, test_g_outputs_l1,
                           cv_stack, n_valid=self.n_valid[1], n_era=self.n_era[1], cv_seed=self.cv_seed,
                           n_epoch=self.n_epoch[1], x_train_reuse=x_tr_reuse, x_test_reuse=x_te_reuse)

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
        #     = self.stacker(models_initializer_l3, self.layers_param[1], x_outputs_l2, self.y_train,
        #                    self.w_train, self.e_train, x_g_outputs_l2, test_outputs_l2, test_g_outputs_l2,
        #                    cv_stack, n_valid=self.n_valid[2], n_era=self.n_era[2], cv_seed=self.cv_seed,
        #                    n_epoch=self.n_epoch[2])
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


# Iterator for stack tree
class StackLayer:

    def __init__(self, models_initializer, stacker, params, x_train, y_train, w_train,
                 e_train, x_g_train, x_test, x_g_test, id_test, cv, n_valid=4, n_era=20, cv_seed=None,
                 input_layer=None, i_layer=1, n_epoch=1, x_train_reuse=None, x_test_reuse=None, dnn_param=None,
                 pred_path=None, stack_output_path=None):

        self.models_initializer = models_initializer
        self.stacker = stacker
        self.params = params
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = w_train
        self.e_train = e_train
        self.x_g_train = x_g_train
        self.x_test = x_test
        self.x_g_test = x_g_test
        self.g_train = x_g_train[:, -1]
        self.g_test = x_g_test[:, -1]
        self.cv = cv
        self.n_valid = n_valid
        self.n_era = n_era
        self.cv_seed = cv_seed
        self.input_layer = input_layer
        self.i_layer = i_layer
        self.n_epoch = n_epoch
        self.x_train_reuse = x_train_reuse
        self.x_test_reuse = x_test_reuse
        self.dnn_param = dnn_param
        self.id_test = id_test
        self.pred_path = pred_path
        self.stack_output_path = stack_output_path

    def save_predict(self, pred_path, test_outputs):

        test_prob = np.mean(test_outputs, axis=1)

        utils.save_pred_to_csv(pred_path, self.id_test, test_prob)

    def train(self, i_epoch=1):

        print('======================================================')
        print('Start Training - Layer: {} | Epoch: {}'.format(self.i_layer, i_epoch))

        if self.i_layer != 1:

            blender_x_tree = np.array([])
            blender_test_tree = np.array([])

            for epoch in range(self.n_epoch):

                epoch_start_time = time.time()

                # Training super layer
                blender_x_e, blender_test_e, _, _ = self.input_layer.train(epoch + 1)

                # Print Shape
                print('------------------------------------------------------')
                print('blender_x_e shape:{}'.format(blender_x_e.shape))
                print('blender_test_e shape:{}'.format(blender_test_e.shape))
                print('------------------------------------------------------')

                if epoch == 0:
                    blender_x_tree = blender_x_e
                    blender_test_tree = blender_test_e
                else:
                    # n_sample * (n_model x n_epoch)
                    blender_x_tree = np.concatenate((blender_x_tree, blender_x_e), axis=1)
                    # n_test_sample * (n_model x n_epoch)
                    blender_test_tree = np.concatenate((blender_test_tree, blender_test_e), axis=1)

                # TODO
                print(blender_test_tree[:, 2])

                epoch_time = time.time() - epoch_start_time
                print('------------------------------------------------------')
                print('Epoch Done!')
                print('Epoch Time: {}s'.format(epoch_time))
                print('======================================================')

                # Save predicted test prob
                self.save_predict(self.pred_path + 'epochs_results/stack_l{}_e{}_'.format(self.i_layer, epoch+1),
                                  blender_test_tree)

                # Stack Group Features
            print('------------------------------------------------------')
            print('Stacking Group Features...')
            blender_x_g_tree = np.column_stack((blender_x_tree, self.g_train))
            blender_test_g_tree = np.column_stack((blender_test_tree, self.g_test))

            # Print Shape
            print('------------------------------------------------------')
            print('blender_x_tree:{}'.format(blender_x_tree.shape))
            print('blender_test_tree:{}'.format(blender_test_tree.shape))
            print('blender_x_g_tree:{}'.format(blender_x_g_tree.shape))
            print('blender_test_g_tree:{}'.format(blender_test_g_tree.shape))
            print('======================================================')

            # Save layer outputs
            utils.save_stack_outputs(self.stack_output_path + 'l{}_'.format(self.i_layer),
                                     blender_x_tree, blender_test_tree, blender_x_g_tree, blender_test_g_tree)

        else:

            blender_x_tree = self.x_train
            blender_test_tree = self.x_g_train
            blender_x_g_tree = self.x_test
            blender_test_g_tree = self.x_g_test

        # Training Stacker
        blender_x_outputs, blender_test_outputs, blender_x_g_outputs, blender_test_g_outputs \
            = self.stacker(self.models_initializer, self.params, blender_x_tree, self.y_train, self.w_train,
                           self.e_train, blender_test_tree, blender_x_g_tree, blender_test_g_tree,
                           self.g_train, self.g_test, cv=self.cv, n_valid=self.n_valid,
                           n_era=self.n_era, cv_seed=self.cv_seed, i_layer=self.i_layer, i_epoch=i_epoch,
                           x_train_reuse=self.x_train_reuse, x_test_reuse=self.x_test_reuse,
                           dnn_param=self.dnn_param)

        return blender_x_outputs, blender_test_outputs, blender_x_g_outputs, blender_test_g_outputs


# Stack Tree
class StackTree:

    id_test = np.array([])
    pred_path = ''
    stack_output_path = ''

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, x_tr_g, x_te_g,
                 pred_path, loss_log_path, stack_output_path, hyper_params, layers_param):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_g_train = x_tr_g
        self.x_g_test = x_te_g
        self.pred_path = pred_path
        self.loss_log_path = loss_log_path
        self.stack_output_path = stack_output_path
        self.layers_param = layers_param
        self.g_train = x_tr_g[:,-1]
        self.g_test = x_te_g[:, -1]
        self.n_valid = hyper_params['n_valid']
        self.n_era = hyper_params['n_era']
        self.n_epoch = hyper_params['n_epoch']
        self.cv_seed = hyper_params['cv_seed']

    def init_models_layer1(self, dnn_l1_params=None):

        LGB_L1 = models.LightGBM(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.id_test, self.x_g_train, self.x_g_test)
        XGB_L1 = models.XGBoost(self.x_train, self.y_train, self.w_train,
                                self.e_train, self.x_test, self.id_test)
        # AB_L1 = models.AdaBoost(self.x_train, self.y_train, self.w_train,
        #                         self.e_train, self.x_test, self.id_test)
        # RF_L1 = models.RandomForest(self.x_train, self.y_train, self.w_train,
        #                             self.e_train, self.x_test, self.id_test)
        # ET_L1 = models.ExtraTrees(self.x_train, self.y_train, self.w_train,
        #                           self.e_train, self.x_test, self.id_test)
        # GB_L1 = models.GradientBoosting(self.x_train, self.y_train, self.w_train,
        #                                 self.e_train, self.x_test, self.id_test)
        DNN_L1 = models.DeepNeuralNetworks(self.x_train, self.y_train, self.w_train,
                                           self.e_train, self.x_test, self.id_test, dnn_l1_params)

        models_l1 = [
                     # LGB_L1,
                     # XGB_L1,
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

    @staticmethod
    def train_models(models_blender, params, x_train, y_train, w_train, x_g_train,
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
                e_train_inputs, x_g_train_inputs, x_test, x_g_test, g_train, g_test, 
                cv, n_valid=4, n_era=20, cv_seed=None, i_layer=1, i_epoch=1,
                x_train_reuse=None, x_test_reuse=None, dnn_param=None):

        if n_era%n_valid != 0:
            assert ValueError('n_era must be an integer multiple of n_valid!')

        # Stack Reused Features
        if x_train_reuse is not None:
            print('------------------------------------------------------')
            print('Stacking Reused Features...')
            x_train_inputs = np.concatenate((x_train_inputs, x_train_reuse), axis=1)  # n_sample * (n_feature + n_reuse)
            x_train_group = x_g_train_inputs[:, -1]
            x_g_train_inputs = np.column_stack((x_train_inputs, x_train_group))       # n_sample * (n_feature + n_reuse + 1)

        if x_test_reuse is not None:
            print('------------------------------------------------------')
            print('Stacking Reused Features...')
            x_test = np.concatenate((x_test, x_test_reuse), axis=1)  # n_sample * (n_feature + n_reuse)
            x_test_group = x_g_test[:, -1]
            x_g_test = np.column_stack((x_test, x_test_group))       # n_sample * (n_feature + n_reuse + 1)

        # Print Shape
        print('------------------------------------------------------')
        print('x_train_inputs shape:{}'.format(x_train_inputs.shape))
        print('x_test shape:{}'.format(x_test.shape))
        print('------------------------------------------------------')

        # Init models blender
        if dnn_param is not None:
            models_blender = models_initializer(dnn_param)
        else:
            models_blender = models_initializer()
        n_model = len(models_blender)

        blender_valid = np.array([])
        blender_test = np.array([])
        # blender_losses = np.array([])
        
        n_cv = int(n_era // n_valid)
        counter_cv = 0

        for x_train, y_train, w_train, x_g_train, x_valid, y_valid, \
            w_valid, x_g_valid, valid_index, valid_era in cv.era_k_fold_for_stack(x=x_train_inputs,
                                                                                  y=y_train_inputs,
                                                                                  w=w_train_inputs,
                                                                                  e=e_train_inputs,
                                                                                  x_g=x_g_train_inputs,
                                                                                  n_valid=n_valid,
                                                                                  n_cv=n_cv,
                                                                                  seed=cv_seed):

            # TODO:
            print(x_train, y_train)
            print(x_train.shape, y_train.shape)

            counter_cv += 1

            print('======================================================')
            print('Layer: {} | Epoch: {} | CV: {}/{}'.format(i_layer, i_epoch, counter_cv, n_cv))
            print('Validation Set Era: ', valid_era)

            # Training on models and get blenders of one cross validation set
            blender_valid_cv, blender_test_cv, \
                blender_losses_cv = self.train_models(models_blender, params, x_train, y_train, w_train,
                                                      x_g_train, x_valid, y_valid, w_valid, x_g_valid,
                                                      valid_index, x_test, x_g_test)

            # TODO
            print(blender_valid_cv.shape, blender_test_cv.shape)

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

            # TODO:
            print(blender_test)
            print(blender_test.shape)

        # Print Shape
        print('======================================================')
        print('blender_valid shape:{}'.format(blender_valid.shape))
        print('blender_test shape:{}'.format(blender_test.shape))

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
        blender_x_outputs = blender_valid_sorted.transpose()      # n_sample * n_model
        blender_test_outputs = blender_test_mean.transpose()      # n_test_sample * n_model

        # TODO
        print(blender_x_outputs.shape, g_train.shape)

        # Stack Group Features
        print('------------------------------------------------------')
        print('Stacking Group Features...')
        blender_x_g_outputs = np.column_stack((blender_x_outputs, g_train))
        blender_test_g_outputs = np.column_stack((blender_test_outputs, g_test))

        # Print Shape
        print('------------------------------------------------------')
        print('blender_x_outputs:{}'.format(blender_x_outputs.shape))
        print('blender_test_outputs:{}'.format(blender_test_outputs.shape))
        print('blender_x_g_outputs:{}'.format(blender_x_g_outputs.shape))
        print('blender_test_g_outputs:{}'.format(blender_test_g_outputs.shape))
        print('======================================================')

        return blender_x_outputs, blender_test_outputs, blender_x_g_outputs, blender_test_g_outputs

    def save_predict(self, pred_path, test_outputs):

        test_prob = np.mean(test_outputs, axis=1)

        utils.save_pred_to_csv(pred_path, self.id_test, test_prob)

    def stack_outputs_train(self, model_name, params, n_valid, n_cv, x_outputs,
                            test_outputs, x_g_outputs, test_g_outputs):

        if model_name == 'LGB':
            model = models.LightGBM(x_outputs, self.y_train,  self.w_train,  self.e_train,
                                    test_outputs,  self.id_test, x_g_outputs, test_g_outputs)
        elif model_name == 'DNN':
            model = models.DeepNeuralNetworks(x_outputs, self.y_train,  self.w_train,  self.e_train,
                                              test_outputs,  self.id_test, params)
        else:
            model = 0
            assert ValueError('Wrong model name!')

        print('Start training {}...'.format(model_name))

        model.train_sklearn(self.pred_path + 'stack_outputs/',  self.loss_log_path,
                            n_valid=n_valid, n_cv=n_cv, parameters=params)

    def stack(self):

        start_time = time.time()

        # Create a CV
        cv_stack = models.CrossValidation()

        # Initializing models for every layer
        models_initializer_l1 = self.init_models_layer1
        models_initializer_l2 = self.init_models_layer2

        # Parameters for DNN models
        dnn_l1_params = self.layers_param[0][-1]
        # dnn_l2_params = self.layers_param[1][-1]
        # dnn_l3_params = self.layers_param[2][-1]

        # Reused features
        x_train_reuse_l2 = self.x_train[:, :88]
        x_test_reuse_l2 = self.x_test[:, :88]

        print('======================================================')
        print('Start training...')

        # Building Graph

        # Layer 1
        stk_l1 = StackLayer(models_initializer_l1, self.stacker, self.layers_param[0], self.x_train, self.y_train,
                            self.w_train, self.e_train, self.x_g_train, self.x_test, self.x_g_test, self.id_test,
                            cv_stack, n_valid=self.n_valid[0], n_era=self.n_era[0], cv_seed=self.cv_seed,
                            i_layer=1, n_epoch=self.n_epoch[0],dnn_param=dnn_l1_params,
                            pred_path=self.pred_path, stack_output_path=self.stack_output_path)

        # Layer 2
        stk_l2 = StackLayer(models_initializer_l2, self.stacker, self.layers_param[1], self.x_train, self.y_train,
                            self.w_train, self.e_train, self.x_g_train, self.x_test, self.x_g_test, self.id_test,
                            cv_stack, n_valid=self.n_valid[1], n_era=self.n_era[1], cv_seed=self.cv_seed,
                            input_layer=stk_l1, i_layer=2, n_epoch=self.n_epoch[1],
                            x_train_reuse=x_train_reuse_l2, x_test_reuse=x_test_reuse_l2,
                            pred_path=self.pred_path, stack_output_path=self.stack_output_path)

        # Training
        _, test_outputs, _, _ = stk_l2.train()

        # Save predicted test prob
        self.save_predict(self.pred_path + 'final_results/stack_', test_outputs)

        total_time = time.time() - start_time
        print('======================================================')
        print('Training Done!')
        print('Total Time: {}s'.format(total_time))
        print('======================================================')


if __name__ == '__main__':

    pass
