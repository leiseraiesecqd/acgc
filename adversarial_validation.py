import time
import utils
import random
import preprocess
import numpy as np
import tensorflow as tf

gan_prob_path = './data/gan_outputs/'

class AdversarialValidation(object):
    """
        Generate Adversarial Validation Set Using GAN
    """

    def __init__(self, x_tr, x_te, parameters):

        self.x_train = x_tr
        self.x_test = x_te
        self.learning_rate = parameters['learning_rate']
        self.epochs = parameters['epochs']
        self.n_discriminator_units = parameters['n_discriminator_units']
        self.n_generator_units = parameters['n_generator_units']
        self.z_dim = parameters['z_dim']
        self.beta1 = parameters['beta1']
        self.batch_size = parameters['batch_size']
        self.keep_prob = parameters['keep_prob']
        self.display_step = parameters['display_step']
        self.train_seed = parameters['train_seed']

        np.random.seed(self.train_seed)

    def model_inputs(self):
        """
            Create the model inputs
        """
        n_feature = self.x_train.shape[1]
        inputs_real = tf.placeholder(tf.float64, (None, n_feature), name='inputs_real')
        inputs_z = tf.placeholder(tf.float64, (None, self.z_dim), name='inputs_z')
        keep_prob = tf.placeholder(tf.float64, name='keep_prob')

        return inputs_real, inputs_z, keep_prob

    def discriminator(self, features, keep_prob, is_training=True, reuse=False):
        """
            Create the discriminator network
        """
        # x_shape = features.get_shape().as_list()
        # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
        # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64, seed=self.train_seed)
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
                                                          biases_initializer=tf.zeros_initializer(dtype=tf.float64))

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
                                                          biases_initializer=tf.zeros_initializer(dtype=tf.float64))

                tf.summary.histogram('fc_layer', layer)

                layer = tf.nn.dropout(layer, keep_prob)

            return layer

        with tf.variable_scope('discriminator', reuse=reuse):

            fc = [features]

            for i in range(len(self.n_discriminator_units)):

                fc.append(fc_layer(fc[i], 'fc{}'.format(i + 1), self.n_discriminator_units[i]))

            outputs = fc[len(self.n_discriminator_units)]

            logits = output_layer(outputs, 'output', 1)

        return outputs, logits

    def generator(self, z, keep_prob, is_training=True):
        """
            Create the generator network
        """
        # x_shape = features.get_shape().as_list()
        # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
        # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64, seed=self.train_seed)
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
                                                          biases_initializer=tf.zeros_initializer(dtype=tf.float64))

                tf.summary.histogram('fc_layer', layer)

                layer = tf.nn.dropout(layer, keep_prob)

            return layer

        with tf.variable_scope('generator', reuse=not is_training):

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
        g_outputs = self.generator(inputs_z, keep_prob, is_training=True)
        d_outputs_real, d_logits_real = self.discriminator(inputs_real, keep_prob, is_training=True, reuse=False)
        d_outputs_fake, d_logits_fake = self.discriminator(g_outputs, keep_prob, is_training=True, reuse=True)

        d_labels_real = tf.ones_like(d_outputs_real) * (1 - 0.1) + np.random.uniform(-0.05, 0.05)
        d_labels_fake = tf.zeros_like(d_outputs_fake) + np.random.uniform(0.0, 0.1)
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

        _, d_logits_similarity = self.discriminator(inputs_real, keep_prob, is_training=False, reuse=True)

        similarities = tf.nn.sigmoid(d_logits_similarity, name='similarities')

        return similarities

    @staticmethod
    def get_batches(x, batch_size):

        n_batches = len(x) // batch_size

        for ii in range(0, n_batches * batch_size + 1, batch_size):

            if ii != n_batches * batch_size - 1:
                batch_x = x[ii: ii + batch_size]
            else:
                batch_x = x[ii:]

            yield batch_x

    def train(self, gan_prob_path=None, global_epochs=1):
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

        batch_counter = 0
        similarity_prob_total = []

        with tf.Session(graph=train_graph) as sess:

            start_time = time.time()

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
                        batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                        # Run optimizers
                        _ = sess.run(d_train_opt, feed_dict={inputs_real: x_batch,
                                                             inputs_z: batch_z,
                                                             keep_prob: self.keep_prob})
                        _ = sess.run(g_train_opt, feed_dict={inputs_real: x_batch,
                                                             inputs_z: batch_z,
                                                             keep_prob: self.keep_prob})

                        if batch_counter % self.display_step == 0 and batch_i > 0:

                            # At losses
                            d_cost = d_loss.eval({inputs_real: x_batch, inputs_z: batch_z, keep_prob: 1.0})
                            g_cost = g_loss.eval({inputs_z: batch_z, keep_prob: 1.0})

                            total_time = time.time() - start_time

                            print('Global_Epoch: {}/{} |'.format(global_epoch_i+1, global_epochs),
                                  'Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                                  'Batch: {:>5} |'.format(batch_counter),
                                  'Time: {:>3.2f}s |'.format(total_time),
                                  'd_Loss: {:.8f} |'.format(d_cost),
                                  'g_Loss: {:.8f}'.format(g_cost))

                print('------------------------------------------------------')
                print('Calculating Similarities of Train Set...')
                similarity_prob = \
                    sess.run(similarities, feed_dict={inputs_real: self.x_train, keep_prob: self.keep_prob})

                similarity_prob_total.append(similarity_prob)

            print('======================================================')
            print('Calculating Final Similarities of Train Set...')
            similarity_prob_mean = np.mean(np.array(similarity_prob_total), axis=0)

            utils.save_np_to_pkl(similarity_prob_mean, gan_prob_path + 'similarity_prob.p')

            return similarity_prob_mean


def generate_validation_set(x_train, x_test, train_seed=None):

    utils.check_dir([gan_prob_path])

    if train_seed is None:
        train_seed = random.randint(0, 500)

    parameters = {'learning_rate': 0.0001,
                  'epochs': 30,
                  'n_discriminator_units': [64, 32, 16],
                  'n_generator_units': [32, 64, 128],
                  'z_dim': 16,
                  'beta1': 0.5,
                  'batch_size': 128,
                  'keep_prob': 0.75,
                  'display_step': 100,
                  'train_seed': train_seed}

    AV = AdversarialValidation(x_train, x_test, parameters)

    similarity_prob = AV.train(gan_prob_path=gan_prob_path, global_epochs=10)

    return similarity_prob


if __name__ == '__main__':

    print('======================================================')
    print('Start Training...')

    start_time = time.time()

    global_train_seed = random.randint(0, 300)

    _x_train, _, _, _, _x_test, _ = utils.load_preprocessed_pd_data(preprocess.preprocessed_path)

    _ = generate_validation_set(_x_train, _x_test, global_train_seed)

    print('======================================================')
    print('All Tasks Done!')
    print('Global Train Seed: {}'.format(global_train_seed))
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')
