import tensorflow_addons as tfa

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize, pixel_shuffle

import tensorflow as tf


def wdsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def wdsr_b(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None):
    print("Created wdsr_b model")
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b)

class NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self):
        self.rgb_mean = tf.constant([0.4488, 0.4371, 0.4040]) * 255
        super(NormalizeLayer, self).__init__()

    def call(self, inputs):
        return (inputs - self.rgb_mean) / 127.5

    def get_config(self):
        return {'rgb_mean': self.rgb_mean}

class NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self):
        self.rgb_mean = tf.constant([0.4488, 0.4371, 0.4040]) * 255
        super(NormalizeLayer, self).__init__()

    def call(self, inputs):
        return (inputs - self.rgb_mean) / 127.5

    def get_config(self):
        return {'rgb_mean': self.rgb_mean}

class DenormalizeLayer(tf.keras.layers.Layer):
    def __init__(self):
        self.rgb_mean = tf.constant([0.4488, 0.4371, 0.4040]) * 255
        super(DenormalizeLayer, self).__init__()

    def call(self, inputs):
        return inputs * 127.5 + self.rgb_mean

    def get_config(self):
        return {'rgb_mean': self.rgb_mean}

class PixelShuffleLayer(tf.keras.layers.Layer):
    def __init__(self, scale):
        self.scale = scale
        super(PixelShuffleLayer, self).__init__()

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        return {'scale': self.scale}

class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, scale):
        self.scale = scale
        super(ScalingLayer, self).__init__()

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {'scale': self.scale}


def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):
    x_in = Input(shape=(None, None, 3))

    # x = Lambda(normalize)(x_in)
    x = NormalizeLayer()(x_in)

    # main branch
    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    
    # m = Lambda(pixel_shuffle(scale))(m)
    m = PixelShuffleLayer(scale)(m)

    # skip branch
    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    
    # s = Lambda(pixel_shuffle(scale))(s)
    s = PixelShuffleLayer(scale)(s)

    x = Add()([m, s])

    # x = Lambda(denormalize)(x)
    x = DenormalizeLayer()(x)

    return Model(x_in, x, name="wdsr")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        # x = Lambda(lambda t: t * scaling)(x)
        x = ScalingLayer(scaling)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        # x = Lambda(lambda t: t * scaling)(x)
        x = ScalingLayer(scaling)
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)
