import time
import utils
import random
import numpy as np
import pandas as pd
import tensorflow as tf

gan_prob_path = './data/gan_outputs/'
gan_preprocessed_data_path = './data/gan_preprocessed_data/'
train_csv_path = './inputs/stock_train_data_20171013.csv'
test_csv_path = './inputs/stock_test_data_20171013.csv'


class AdversarialValidation(object):
    """
        Generate Adversarial Validation Set Using GAN
    """

    def __init__(self, parameters=None, train_path=None, test_path=None,
                 gan_preprocess_path=None, load_preprocessed_data=False):

        self.train_path = train_path
        self.test_path = test_path
        self.preprocessed_path = gan_preprocess_path

        self.learning_rate = parameters['learning_rate']
        self.epochs = parameters['epochs']
        self.n_discriminator_units = parameters['n_discriminator_units']
        self.n_generator_units = parameters['n_generator_units']
        self.z_dim = parameters['z_dim']
        self.beta1 = parameters['beta1']
        self.batch_size = parameters['batch_size']
        self.d_epochs = parameters['d_epochs']
        self.g_epochs = parameters['g_epochs']
        self.keep_prob = parameters['keep_prob']
        self.display_step = parameters['display_step']
        self.show_step = parameters['show_step']
        self.train_seed = parameters['train_seed']

        np.random.seed(self.train_seed)

        if load_preprocessed_data is True:

            # Load Preprocessed Data from pickle File
            self.x_train, self.x_test = self.load_data_from_pickle()

        else:

            # Load Data
            self.x_train, self.x_test, self.g_train, self.g_test = self.load_data_from_csv()

            # Drop outliers
            self.drop_test_outliers_by_value()

            # Min Max Scale
            self.min_max_scale()

            # Convert column 'group' to dummies
            self.convert_group_to_dummies()

            # Convert pandas DataFrame to numpy array
            self.convert_pd_to_np()

            # Save Preprocessed Data to pickle File
            self.save_data()

    def load_data_from_pickle(self):

        print('Loading Preprocessed Data...')

        x_train = utils.load_pkl_to_np(self.preprocessed_path + 'x_train_gan.p')
        x_test = utils.load_pkl_to_np(self.preprocessed_path + 'x_test_gan.p')

        return x_train, x_test

    def load_data_from_csv(self):

        try:
            print('Loading data...')
            train_f = pd.read_csv(self.train_path, header=0, dtype=np.float64)
            test_f = pd.read_csv(self.test_path, header=0, dtype=np.float64)
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        # Drop Unnecessary Columns
        # self.x_train = train_f.drop(['id', 'weight', 'label', 'group', 'era'], axis=1)
        x_train = train_f.drop(['id', 'weight', 'label', 'group', 'era', 'feature43'], axis=1)
        x_test = test_f.drop(['id', 'group', 'feature43'], axis=1)
        g_train = train_f['group']
        g_test = test_f['group']

        return x_train, x_test, g_train, g_test

    def drop_test_feature_outliers_by_value(self, feature, upper_test=None, lower_test=None):

        # Drop upper outliers in self.x_train
        if upper_test is not None:
            self.x_test[feature].loc[self.x_test[feature] > upper_test] = upper_test

        # Drop lower outlines in self.x_train
        if lower_test is not None:
            self.x_test[feature].loc[self.x_test[feature] < lower_test] = lower_test

    def drop_train_feature_outliers_by_value(self, feature, upper_train=None, lower_train=None):

        # Drop upper outliers in self.x_train
        if upper_train is not None:
            self.x_train[feature].loc[self.x_test[feature] > upper_train] = upper_train

        # Drop lower outlines in self.x_train
        if lower_train is not None:
            self.x_train[feature].loc[self.x_test[feature] < lower_train] = lower_train

    def drop_test_outliers_by_value(self):

        print('Dropping outliers...')

        self.drop_test_feature_outliers_by_value('feature0', None, -3)
        self.drop_test_feature_outliers_by_value('feature1', 4, None)
        self.drop_test_feature_outliers_by_value('feature2', 5, -10)
        self.drop_test_feature_outliers_by_value('feature3', 4, None)
        self.drop_test_feature_outliers_by_value('feature4', 4, None)
        self.drop_test_feature_outliers_by_value('feature5', 3, -3)
        self.drop_test_feature_outliers_by_value('feature6', 3.5, -2)
        self.drop_test_feature_outliers_by_value('feature7', 4, None)
        self.drop_test_feature_outliers_by_value('feature8', 5, -5)
        self.drop_test_feature_outliers_by_value('feature9', 4, None)
        self.drop_test_feature_outliers_by_value('feature10', None, None)
        self.drop_test_feature_outliers_by_value('feature11', 5, None)
        self.drop_test_feature_outliers_by_value('feature12', 2.5, -3)
        self.drop_test_feature_outliers_by_value('feature13', None, None)
        self.drop_test_feature_outliers_by_value('feature14', None, -3.5)
        self.drop_test_feature_outliers_by_value('feature15', 2.5, None)
        self.drop_test_feature_outliers_by_value('feature16', 4, None)
        self.drop_test_feature_outliers_by_value('feature17', 4, None)
        self.drop_test_feature_outliers_by_value('feature18', None, -4)
        self.drop_test_feature_outliers_by_value('feature19', 3, -3)
        self.drop_test_feature_outliers_by_value('feature20', 5, None)
        self.drop_test_feature_outliers_by_value('feature21', 4, -3.5)
        self.drop_test_feature_outliers_by_value('feature22', 3, None)
        self.drop_test_feature_outliers_by_value('feature23', None, None)
        self.drop_test_feature_outliers_by_value('feature24', None, -4)
        self.drop_test_feature_outliers_by_value('feature25', 3.5, None)
        self.drop_test_feature_outliers_by_value('feature26', None, -2.9)
        self.drop_test_feature_outliers_by_value('feature27', 3, None)
        self.drop_test_feature_outliers_by_value('feature28', 4, -4)
        self.drop_test_feature_outliers_by_value('feature29', 4, None)
        self.drop_test_feature_outliers_by_value('feature30', 6, None)
        self.drop_test_feature_outliers_by_value('feature31', 4, None)
        self.drop_test_feature_outliers_by_value('feature32', 5, None)
        self.drop_test_feature_outliers_by_value('feature33', 5, -5)
        self.drop_test_feature_outliers_by_value('feature34', 3.5, -2.5)
        self.drop_test_feature_outliers_by_value('feature35', 5, -3)
        self.drop_test_feature_outliers_by_value('feature36', 6.5, None)
        self.drop_test_feature_outliers_by_value('feature37', 4.5, None)
        self.drop_test_feature_outliers_by_value('feature38', 2.5, -2.5)
        self.drop_test_feature_outliers_by_value('feature39', 2.5, -2.5)
        self.drop_test_feature_outliers_by_value('feature40', None, -3.5)
        self.drop_test_feature_outliers_by_value('feature41', None, None)
        self.drop_test_feature_outliers_by_value('feature42', 4, -2)
        # self.drop_test_feature_outliers_by_value('feature43', None, None)
        self.drop_test_feature_outliers_by_value('feature44', 8, None)
        self.drop_test_feature_outliers_by_value('feature45', 3.5, None)
        self.drop_test_feature_outliers_by_value('feature46', 5, None)
        self.drop_test_feature_outliers_by_value('feature47', None, -3)
        self.drop_test_feature_outliers_by_value('feature48', None, None)
        self.drop_test_feature_outliers_by_value('feature49', 4, -4)
        self.drop_test_feature_outliers_by_value('feature50', None, -4.2)
        self.drop_test_feature_outliers_by_value('feature51', 5, -1.5)
        self.drop_test_feature_outliers_by_value('feature52', 5, None)
        self.drop_test_feature_outliers_by_value('feature53', 3, -3)
        self.drop_test_feature_outliers_by_value('feature54', 3, -3)
        self.drop_test_feature_outliers_by_value('feature55', 4.5, -2)
        self.drop_test_feature_outliers_by_value('feature56', 3.5, None)
        self.drop_test_feature_outliers_by_value('feature57', 3.5, None)
        self.drop_test_feature_outliers_by_value('feature58', 4, None)
        self.drop_test_feature_outliers_by_value('feature59', 4, -2)
        self.drop_test_feature_outliers_by_value('feature60', 4, None)
        self.drop_test_feature_outliers_by_value('feature61', 5, -5)
        self.drop_test_feature_outliers_by_value('feature62', 3, -3.5)
        self.drop_test_feature_outliers_by_value('feature63', 4, -2.4)
        self.drop_test_feature_outliers_by_value('feature64', 4.5, None)
        self.drop_test_feature_outliers_by_value('feature65', 2.5, -3.5)
        self.drop_test_feature_outliers_by_value('feature66', 7.5, -5)
        self.drop_test_feature_outliers_by_value('feature67', None, None)
        self.drop_test_feature_outliers_by_value('feature68', 4.5, None)
        self.drop_test_feature_outliers_by_value('feature69', 3.5, None)
        self.drop_test_feature_outliers_by_value('feature70', None, -4.5)
        self.drop_test_feature_outliers_by_value('feature71', 5, -2.2)
        self.drop_test_feature_outliers_by_value('feature72', 5, -5)
        self.drop_test_feature_outliers_by_value('feature73', None, None)
        self.drop_test_feature_outliers_by_value('feature74', 4, None)
        self.drop_test_feature_outliers_by_value('feature75', 3, -3.5)
        self.drop_test_feature_outliers_by_value('feature76', 6, None)
        self.drop_test_feature_outliers_by_value('feature77', None, None)
        self.drop_test_feature_outliers_by_value('feature78', None, -3.5)
        self.drop_test_feature_outliers_by_value('feature79', 4, -2.5)
        self.drop_test_feature_outliers_by_value('feature80', 4.2, None)
        self.drop_test_feature_outliers_by_value('feature81', 4.8, None)
        self.drop_test_feature_outliers_by_value('feature82', 4.5, None)
        self.drop_test_feature_outliers_by_value('feature83', 3, -2.8)
        self.drop_test_feature_outliers_by_value('feature84', None, -3)
        self.drop_test_feature_outliers_by_value('feature85', 3.6, None)
        self.drop_test_feature_outliers_by_value('feature86', 4.5, None)
        self.drop_test_feature_outliers_by_value('feature87', 7.8, None)

    # Min Max scale
    def min_max_scale(self):

        print('Min Max Scaling Data...')
        x_min = np.zeros(len(self.x_test.columns), dtype=np.float64)
        x_max = np.zeros(len(self.x_test.columns), dtype=np.float64)

        for i, each in enumerate(self.x_test.columns):
            x_max[i], x_min[i] = self.x_test[each].max(), self.x_test[each].min()
            self.x_test.loc[:, each] = (self.x_test[each] - x_min[i]) / (x_max[i] - x_min[i])

        for i, each in enumerate(self.x_train.columns):
            self.x_train.loc[:, each] = (self.x_train[each] - x_min[i]) / (x_max[i] - x_min[i])

        # Drop outliers of train set
        print('Dropping Outliers of Train Set...')
        for i in range(88):
            if i != 43:
                self.drop_test_feature_outliers_by_value('feature' + str(i), 1, 0)

    def convert_group_to_dummies(self):

        print('Converting Groups to Dummies...')

        group_train_dummies = pd.get_dummies(self.g_train, prefix='group')
        self.x_train = self.x_train.join(group_train_dummies)

        group_test_dummies = pd.get_dummies(self.g_test, prefix='group')
        self.x_test = self.x_test.join(group_test_dummies)

    def convert_pd_to_np(self):

        print('Converting pandas DataFrame to numpy array...')

        self.x_train = np.array(self.x_train, dtype=np.float64)
        self.x_test = np.array(self.x_test, dtype=np.float64)

    def save_data(self):

        print('Saving Preprocessed Data...')

        utils.save_np_to_pkl(self.x_train, self.preprocessed_path + 'x_train_gan.p')
        utils.save_np_to_pkl(self.x_test, self.preprocessed_path + 'x_test_gan.p')

    def model_inputs(self):
        """
            Create the model inputs
        """
        n_feature = self.x_train.shape[1]
        inputs_real = tf.placeholder(tf.float32, (None, n_feature), name='inputs_real')
        inputs_z = tf.placeholder(tf.float32, (None, self.z_dim), name='inputs_z')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        return inputs_real, inputs_z, keep_prob

    def discriminator(self, features, keep_prob, is_training=True, reuse=False):
        """
            Create the discriminator network
        """
        # x_shape = features.get_shape().as_list()
        # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
        weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        # weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)
        weights_reg = tf.contrib.layers.l2_regularizer(1e-3)
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {'is_training': is_training}

        def fc_layer(x_tensor, layer_name, num_outputs):

            with tf.name_scope(layer_name):

                layer = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                          num_outputs=num_outputs,
                                                          activation_fn=tf.nn.sigmoid,
                                                          weights_initializer=weights_initializer,
                                                          weights_regularizer=weights_reg,
                                                          normalizer_fn=normalizer_fn,
                                                          normalizer_params=normalizer_params,
                                                          biases_initializer=tf.zeros_initializer(dtype=tf.float32))

                tf.summary.histogram('fc_layer', layer)

                layer = tf.nn.dropout(layer, keep_prob)

            return layer

        def output_layer(x_tensor, layer_name, num_outputs):

            with tf.name_scope(layer_name):

                layer = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                          num_outputs=num_outputs,
                                                          activation_fn=None,
                                                          weights_initializer=weights_initializer,
                                                          weights_regularizer=weights_reg,
                                                          normalizer_fn=normalizer_fn,
                                                          normalizer_params=normalizer_params,
                                                          biases_initializer=tf.zeros_initializer(dtype=tf.float32))

                tf.summary.histogram('fc_layer', layer)

                layer = tf.nn.dropout(layer, keep_prob)

            return layer

        with tf.variable_scope('discriminator', reuse=reuse):

            fc = [features]

            for i in range(len(self.n_discriminator_units)):

                fc.append(fc_layer(fc[i], 'fc{}'.format(i + 1), self.n_discriminator_units[i]))

            fc_final = fc[len(self.n_discriminator_units)]

            logits = output_layer(fc_final, 'output', 1)

            outputs = tf.sigmoid(logits)

        return logits, outputs

    def generator(self, z, keep_prob, is_training=True, reuse=False):
        """
            Create the generator network
        """
        # x_shape = features.get_shape().as_list()
        # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
        weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        # weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)
        weights_reg = tf.contrib.layers.l2_regularizer(1e-3)
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {'is_training': is_training}

        def fc_layer(x_tensor, layer_name, num_outputs):

            with tf.name_scope(layer_name):

                layer = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                          num_outputs=num_outputs,
                                                          activation_fn=tf.nn.sigmoid,
                                                          weights_initializer=weights_initializer,
                                                          weights_regularizer=weights_reg,
                                                          normalizer_fn=normalizer_fn,
                                                          normalizer_params=normalizer_params,
                                                          biases_initializer=tf.zeros_initializer(dtype=tf.float32))

                tf.summary.histogram('fc_layer', layer)

                layer = tf.nn.dropout(layer, keep_prob)

            return layer

        with tf.variable_scope('generator', reuse=reuse):

            fc = [z]

            for i in range(len(self.n_generator_units)):
                fc.append(fc_layer(fc[i], 'fc{}'.format(i + 1), self.n_generator_units[i]))

            fc.append(fc_layer(fc[-1], 'fc{}'.format(len(fc)), self.x_test.shape[1]))

            outputs = fc[-1]

        return outputs

    def model_loss(self, inputs_real, inputs_z, keep_prob):
        """
            Get the loss for the discriminator and generator
        """
        g_outputs = self.generator(inputs_z, keep_prob, is_training=True, reuse=False)
        d_logits_real, d_outputs_real = self.discriminator(inputs_real, keep_prob, is_training=True, reuse=False)
        d_logits_fake, d_outputs_fake = self.discriminator(g_outputs, keep_prob, is_training=True, reuse=True)

        #  d_labels_real = tf.ones_like(d_outputs_real) * (1 - 0.1) + np.random.uniform(-0.05, 0.05)
        #  d_labels_fake = tf.zeros_like(d_outputs_fake) + np.random.uniform(0.0, 0.1)
        #  g_labels = tf.ones_like(d_outputs_fake)
        d_labels_real = tf.ones_like(d_outputs_real)
        d_labels_fake = tf.zeros_like(d_outputs_fake)
        g_labels = tf.ones_like(d_outputs_fake)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                             labels=d_labels_real))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                             labels=d_labels_fake))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=g_labels))

        d_loss = d_loss_real + d_loss_fake

        return d_loss, g_loss

    def model_opt(self, d_loss, g_loss):
        """
            Get optimization operations
        """
        # Get variables
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt

    def get_similarity(self, inputs_real, keep_prob):

        _, similarity_prob = self.discriminator(inputs_real, keep_prob, is_training=False, reuse=True)

        return similarity_prob

    def get_generator(self, inputs_z, keep_prob):

        g_outputs = self.generator(inputs_z, keep_prob, is_training=False, reuse=True)

        return g_outputs

    @staticmethod
    def get_batches(x, batch_size):

        n_batches = len(x) // batch_size

        for ii in range(0, n_batches * batch_size + 1, batch_size):

            if ii != n_batches * batch_size - 1:
                batch_x = x[ii: ii + batch_size]
            else:
                batch_x = x[ii:]

            yield batch_x

    def train(self, similarity_prob_path=None, global_epochs=1, return_similarity_prob=False):
        """
            Train the GAN
        """
        print('======================================================')
        print('Training GAN for Adversarial Validation Set...')
        print('------------------------------------------------------')

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Get inputs
            inputs_real, inputs_z, keep_prob = self.model_inputs()

            # Get losses
            d_loss, g_loss = self.model_loss(inputs_real, inputs_z, keep_prob)

            # Get optimizers
            d_train_opt, g_train_opt = self.model_opt(d_loss, g_loss)

            # Get similarities
            similarities = self.get_similarity(inputs_real, keep_prob)

            # Get generator
            g_outputs = self.get_generator(inputs_z, keep_prob)

        batch_counter = 0
        similarity_prob_total = []

        with tf.Session(graph=train_graph) as sess:

            local_start_time = time.time()

            for global_epoch_i in range(global_epochs):

                print('======================================================')
                print('Training on Global Epoch: {}/{}'.format(global_epoch_i+1, global_epochs))
                print('------------------------------------------------------')

                x_test = self.x_test
                np.random.shuffle(x_test)

                sess.run(tf.global_variables_initializer())

                for epoch_i in range(self.epochs):

                    for batch_i, x_batch in enumerate(self.get_batches(x_test, self.batch_size)):

                        batch_counter += 1

                        # Sample random noise
                        batch_z = np.random.uniform(0, 1, size=(self.batch_size, self.z_dim))

                        # Run optimizers
                        for _ in range(self.d_epochs):
                            sess.run(d_train_opt, feed_dict={inputs_real: x_batch,
                                                             inputs_z: batch_z,
                                                             keep_prob: self.keep_prob})
                        for _ in range(self.g_epochs):
                            sess.run(g_train_opt, feed_dict={inputs_real: x_batch,
                                                             inputs_z: batch_z,
                                                             keep_prob: self.keep_prob})

                        if batch_counter % self.display_step == 0 and batch_i > 0:

                            # At losses
                            d_cost = d_loss.eval({inputs_real: x_batch, inputs_z: batch_z, keep_prob: 1.0})
                            g_cost = g_loss.eval({inputs_z: batch_z, keep_prob: 1.0})

                            total_time = time.time() - local_start_time

                            print('Global_Epoch: {}/{} |'.format(global_epoch_i+1, global_epochs),
                                  'Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                                  'Batch: {:>5} |'.format(batch_counter),
                                  'Time: {:>3.2f}s |'.format(total_time),
                                  'd_Loss: {:.8f} |'.format(d_cost),
                                  'g_Loss: {:.8f}'.format(g_cost))

                        if batch_counter % self.show_step == 0 and batch_i > 0:

                            example_z = np.random.uniform(0, 1, size=(self.batch_size, self.z_dim))

                            # At losses
                            generator_outputs = sess.run(g_outputs, feed_dict={inputs_z: example_z, keep_prob: 1.0})
                            discriminator_outputs = \
                                sess.run(similarities, feed_dict={inputs_real: generator_outputs, keep_prob: 1.0})

                            print('------------------------------------------------------')
                            print('Generator Outputs:\n', generator_outputs[0])
                            print('------------------------------------------------------')
                            print('Discriminator Outputs:\n', discriminator_outputs[:50].reshape(1, -1))
                            print('------------------------------------------------------')
                            
                print('------------------------------------------------------')
                print('Calculating Similarities of Train Set...')
                similarity_prob = \
                    sess.run(similarities, feed_dict={inputs_real: self.x_train, keep_prob: 1.0})

                similarity_prob_total.append(similarity_prob)

            print('======================================================')
            print('Calculating Final Similarities of Train Set...')
            similarity_prob_mean = np.mean(np.array(similarity_prob_total), axis=0)

            utils.save_np_to_pkl(similarity_prob_mean, similarity_prob_path + 'similarity_prob.p')

            if return_similarity_prob is True:
                return similarity_prob_mean


def generate_validation_set(train_path=None, test_path=None, similarity_prob_path=None,
                            load_preprocessed_data=False, gan_preprocess_path=None,
                            train_seed=None, global_epochs=1, return_similarity_prob=False):

    if train_seed is None:
        train_seed = random.randint(0, 500)

    parameters = {'learning_rate': 0.001,
                  'epochs': 30,
                  'n_discriminator_units': [64, 32, 16],
                  'n_generator_units': [32, 48, 72],
                  'z_dim': 24,
                  'beta1': 0.9,
                  'batch_size': 16,
                  'd_epochs': 1,
                  'g_epochs': 1,
                  'keep_prob': 0.9,
                  'display_step': 100,
                  'show_step': 2000,
                  'train_seed': train_seed}

    AV = AdversarialValidation(parameters=parameters, train_path=train_path,
                               load_preprocessed_data=load_preprocessed_data,
                               test_path=test_path, gan_preprocess_path=gan_preprocess_path)

    if return_similarity_prob is True:
        similarity_prob = AV.train(similarity_prob_path=similarity_prob_path, global_epochs=global_epochs,
                                   return_similarity_prob=return_similarity_prob)
        return similarity_prob
    else:
        AV.train(similarity_prob_path=similarity_prob_path, global_epochs=global_epochs,
                 return_similarity_prob=return_similarity_prob)


if __name__ == '__main__':

    print('======================================================')
    print('Start Training...')

    start_time = time.time()

    utils.check_dir([gan_prob_path, gan_preprocessed_data_path])

    global_train_seed = random.randint(0, 500)

    generate_validation_set(train_path=train_csv_path, test_path=test_csv_path,
                            similarity_prob_path=gan_prob_path,
                            # load_preprocessed_data=True,
                            gan_preprocess_path=gan_preprocessed_data_path,
                            train_seed=global_train_seed, global_epochs=1)

    print('======================================================')
    print('All Tasks Done!')
    print('Global Train Seed: {}'.format(global_train_seed))
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')
