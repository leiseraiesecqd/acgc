import numpy as np
import tensorflow as tf


class GenerateValidationSet(object):

    def __init__(self, x_tr, x_te):

        self.x_train = x_tr
        self.x_test = x_te
        self.z_dim = 20

    def model_inputs(self):
        """
            Create the model inputs
        """
        n_feature = self.x_train.shape[1]
        inputs_real = tf.placeholder(tf.float32, (None, n_feature), name='inputs_real')
        inputs_z = tf.placeholder(tf.float32, (None, self.z_dim), name='inputs_z')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        return inputs_real, inputs_z, learning_rate

    def discriminator(images, reuse=False):
        """
            Create the discriminator network
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            alpha = 0.2
            keep_prob = 0.9

            # Input layer
            x1 = tf.layers.conv2d(images, 64, 5, strides=2,
                                  padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())  # Use xavier initializer
            relu1 = tf.maximum(alpha * x1, x1)
            relu1 = tf.nn.dropout(relu1, keep_prob)

            # Conv layer
            x2 = tf.layers.conv2d(relu1, 128, 5, strides=2,
                                  padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn2 = tf.layers.batch_normalization(x2, training=True)
            relu2 = tf.maximum(alpha * bn2, bn2)
            relu2 = tf.nn.dropout(relu2, keep_prob)

            # Conv layer
            x3 = tf.layers.conv2d(relu2, 256, 5, strides=2,
                                  padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn3 = tf.layers.batch_normalization(x3, training=True)
            relu3 = tf.maximum(alpha * bn3, bn3)
            relu3 = tf.nn.dropout(relu3, keep_prob)

            # Flat layer
            unit_num = relu3.get_shape()[1] * relu3.get_shape()[2] * relu3.get_shape()[3]
            flat = tf.reshape(relu3, (-1, int(unit_num)))

            # Logits
            logits = tf.layers.dense(flat, 1)

            # Output
            output = tf.sigmoid(logits)

        return output, logits