import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class TemperatureSoftmax(Layer):

    def __init__(self, temperature, **kwargs):
        self.t = tf.constant(temperature, dtype=tf.float32)
        super(TemperatureSoftmax, self).__init__(**kwargs)


    def call(self, x, mask=None):
        """
        e ^ (x_i / t) / (sum_i (e^(x_i / t)))

        e ^ (x_i) / (sum (e^(x_i)))

        [0.1, 0.2, 0.2, 0.1, 0.3, 0.1]
        """
        numer = tf.exp(tf.div(x, self.t))
        denom = tf.reduce_sum(numer)
        return tf.div(numer, denom)


    def get_output_shape_for(self, input_shape):
        return input_shape
