import numpy as np
import tensorflow as tf


class AdversarialValidation(object):

    def __init__(self, x_tr, x_te):

        self.x_train = x_tr
        self.x_test = x_te
        self.z_dim = 20
        self.train_seed = 0

    def model_inputs(self):
        """
            Create the model inputs
        """
        n_feature = self.x_train.shape[1]
        inputs_real = tf.placeholder(tf.float32, (None, n_feature), name='inputs_real')
        inputs_z = tf.placeholder(tf.float32, (None, self.z_dim), name='inputs_z')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        return inputs_real, inputs_z, learning_rate

    def discriminator(self, features, keep_prob, is_training):
        """
            Create the discriminator network
        """

        train_seed = self.train_seed
        reuse = not is_training

        # x_shape = features.get_shape().as_list()
        # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
        # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64, seed=train_seed)
        weights_reg = tf.contrib.layers.l2_regularizer(1e-3)
        normalizer_fn = tf.contrib.layers.batch_norm
        normalizer_params = {'is_training': is_training}

        def fc_layer(x_tensor, layer_name, num_outputs):

            with tf.name_scope(layer_name):

                fc = tf.contrib.layers.fully_connected(x_tensor,
                                                       num_outputs,
                                                       activation_fn=tf.nn.sigmoid,
                                                       weights_initializer=weights_initializer,
                                                       weights_regularizer=weights_reg,
                                                       normalizer_fn=normalizer_fn,
                                                       normalizer_params=normalizer_params,
                                                       biases_initializer=tf.zeros_initializer(dtype=tf.float64))

                tf.summary.histogram('fc_layer', fc)

                fc = tf.nn.dropout(fc, keep_prob)

            return fc

        with tf.variable_scope('discriminator', reuse=reuse):

            full_connected = [features]

            for i in range(len(n_unit)):

                fc.append(self.fc_layer(fc[i], 'fc{}'.format(i + 1), n_unit[i], keep_prob, is_training))

            logit_ = self.output_layer(fc[len(n_unit)], 'output', 1)


        return output, logits