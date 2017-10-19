import numpy as np
import tensorflow as tf


class GenerateValidationSet(object):

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])


    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        """
        Create the model inputs
        :param image_width: The input image width
        :param image_height: The input image height
        :param image_channels: The number of image channels
        :param z_dim: The dimension of Z
        :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
        """
        # TODO: Implement Function
        inputs_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='inputs_real')
        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='inputs_z')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        return inputs_real, inputs_z, learning_rate