from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Concatenate, PReLU
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize, pixel_shuffle

import tensorflow as tf

def prelu_activation(initial_val = 0.1):
    return PReLU(alpha_initializer = tf.constant_initializer(value = initial_val), shared_axes=[1, 2])

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs)


def dscn(starting_num_filters = 64, min_filters = 24, filters_decay_gamma = 1.5, nin_filters = 64, n_fe_layers = 8, scale=4):
    
    nin_filters2 = nin_filters // 2

    x_in = Input(shape=(None, None, 3))

    normalised = Lambda(normalize)(x_in)

    # Generate the 1st layer
    # layer_1 = conv2d_weightnorm(num_filters, 3, padding='same', , use_bias=True, activation=prelu_activation())(x_in)
    # layer_1 = conv2d_block(x_in, num_filters, 3, padding='same', initial_alpha=0.1)

    # conv_layers = [layer_1]

    ############################ Build feature extraction layers ###########################

    conv_layers = []
    total_filter_layers = 0
    previous_num_filters = starting_num_filters

    for layer_no in range(n_fe_layers):

        num_filters = int( max( previous_num_filters//filters_decay_gamma, min_filters) )

        # Find the correct input layer
        input_layer = conv_layers[-1] if len(conv_layers) > 0 else normalised

        layer_n = conv2d_weightnorm(num_filters, 3, padding='same', use_bias=True, activation=prelu_activation())(input_layer)

        conv_layers.append( layer_n )
        total_filter_layers += num_filters
        previous_num_filters = num_filters

    feature_extraction_layer = Concatenate(name="Feature_Extraction_Layer")(conv_layers)


    ######################### Build Reconstruction layers ##################################

    A1 = conv2d_weightnorm(nin_filters, 1, padding='same', use_bias=True, activation=prelu_activation())(feature_extraction_layer)

    B1 = conv2d_weightnorm(nin_filters2, 1, padding='same', use_bias=True, activation=prelu_activation())(feature_extraction_layer)

    B2 = conv2d_weightnorm(nin_filters2, 1, padding='same', use_bias=True, activation=prelu_activation())(B1)

    reconstruction_layer = Concatenate(name="Reconstruction_Layer")([A1, B2])

    ######################### Upsampling Layer ############################################

    upsampled_conv = conv2d_weightnorm(3 * (scale**2), 5, padding='same', use_bias=True)(reconstruction_layer)

    upsampled_layer = Lambda(pixel_shuffle(scale))(upsampled_conv)

    ######################## Reconstruction Layer ###########################################

    denormalised = Lambda(denormalize)(upsampled_layer)

    model = Model(x_in, denormalised, name="dscn")
    
    return model


def dscn_bw(starting_num_filters = 64, min_filters = 24, filters_decay_gamma = 1.5, nin_filters = 64, n_fe_layers = 8, scale=4):
    
    nin_filters2 = nin_filters // 2

    x_in = Input(shape=(None, None, 1))

    normalised = Lambda(lambda x: x/255)(x_in)

    # Generate the 1st layer
    # layer_1 = conv2d_weightnorm(num_filters, 3, padding='same', , use_bias=True, activation=prelu_activation())(x_in)
    # layer_1 = conv2d_block(x_in, num_filters, 3, padding='same', initial_alpha=0.1)

    # conv_layers = [layer_1]

    ############################ Build feature extraction layers ###########################

    conv_layers = []
    total_filter_layers = 0
    previous_num_filters = starting_num_filters

    for layer_no in range(n_fe_layers):

        num_filters = int( max( previous_num_filters//filters_decay_gamma, min_filters) )

        # Find the correct input layer
        input_layer = conv_layers[-1] if len(conv_layers) > 0 else normalised

        layer_n = conv2d_weightnorm(num_filters, 3, padding='same', use_bias=True, activation=prelu_activation())(input_layer)

        conv_layers.append( layer_n )
        total_filter_layers += num_filters
        previous_num_filters = num_filters

    feature_extraction_layer = Concatenate(name="Feature_Extraction_Layer")(conv_layers)


    ######################### Build Reconstruction layers ##################################

    A1 = conv2d_weightnorm(nin_filters, 1, padding='same', use_bias=True, activation=prelu_activation())(feature_extraction_layer)

    B1 = conv2d_weightnorm(nin_filters2, 1, padding='same', use_bias=True, activation=prelu_activation())(feature_extraction_layer)

    B2 = conv2d_weightnorm(nin_filters2, 1, padding='same', use_bias=True, activation=prelu_activation())(B1)

    reconstruction_layer = Concatenate(name="Reconstruction_Layer")([A1, B2])

    ######################### Upsampling Layer ############################################

    upsampled_conv = conv2d_weightnorm(3 * (scale**2), 5, padding='same', use_bias=True)(reconstruction_layer)

    upsampled_layer = Lambda(pixel_shuffle(scale))(upsampled_conv)

    ######################## Reconstruction Layer ###########################################

    denormalised = Lambda(lambda x: x*255.0 )(upsampled_layer)

    model = Model(x_in, denormalised, name="dscn")
    
    return model