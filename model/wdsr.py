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

    x = Lambda(normalize)(x_in)
    # x = NormalizeLayer()(x_in)

    # main branch
    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weightnorm(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    
    m = Lambda(pixel_shuffle(scale))(m)
    # m = PixelShuffleLayer(scale)(m)

    # skip branch
    s = conv2d_weightnorm(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    
    s = Lambda(pixel_shuffle(scale))(s)
    # s = PixelShuffleLayer(scale)(s)

    x = Add()([m, s])

    x = Lambda(denormalize)(x)
    # x = DenormalizeLayer()(x)

    return Model(x_in, x, name="wdsr")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
        # x = ScalingLayer(scaling)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
        # x = ScalingLayer(scaling)
    x = Add()([x_in, x])
    return x


from typeguard import typechecked
@tf.keras.utils.register_keras_serializable(package="Addons")
class OurWeightNormalization(tf.keras.layers.Wrapper):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNormalization wrapper works for keras and tf layers.
    ```python
      net = WeightNormalization(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = WeightNormalization(
          tf.keras.layers.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(120, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(n_classes),
          data_init=True)(net)
    ```
    Arguments:
      layer: a layer instance.
      data_init: If `True` use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """

    @typechecked
    def __init__(self, layer: tf.keras.layers, data_init: bool = True, **kwargs):
        super().__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name="layer")
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

        if self.data_init and self.is_rnn:
            logging.warning(
                "WeightNormalization: Using `data_init=True` with RNNs "
                "is advised against by the paper. Use `data_init=False`."
            )

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, "kernel"):
            raise ValueError(
                "`WeightNormalization` must wrap a layer that"
                " contains a `kernel` for weights"
            )

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.g = self.add_weight(
            name="g",
            shape=(self.layer_depth,),
            initializer="ones",
            dtype=kernel.dtype,
            trainable=True,
        )
        self.v = kernel

        self._initialized = self.add_weight(
            name="initialized",
            shape=None,
            initializer="zeros",
            dtype=tf.dtypes.uint8,
            trainable=False,
        )

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope("data_dep_init"):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config["config"]["trainable"] = False
                self._naked_clone_layer = tf.keras.layers.deserialize(layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if not self.is_rnn:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope("compute_weights"):
            # Replace kernel by normalized weight variable.
            kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * g

            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self, inputs):
        """Initialize weight g.
        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        with tf.control_dependencies(
            [
                # tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                #     self._initialized, False, message="The layer has been initialized."
                # )
            ]
        ):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope("init_norm"):
            v_flat = tf.reshape(self.v, [-1, self.layer_depth])
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        with tf.name_scope("data_dep_init"):
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1.0 / tf.math.sqrt(v_init + 1e-10)

            # RNNs have fused kernels that are tiled
            # Repeat scale_init to match the shape of fused kernel
            # Note: This is only to support the operation,
            # the paper advises against RNN+data_dep_init
            if scale_init.shape[0] != self.g.shape[0]:
                rep = int(self.g.shape[0] / scale_init.shape[0])
                scale_init = tf.tile(scale_init, [rep])

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, "bias") and self.layer.bias is not None:
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]

    def get_config(self):
        config = {"data_init": self.data_init}
        base_config = super().get_config()
        return {**base_config, **config}

    def remove(self):
        kernel = tf.Variable(
            tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g,
            name="recurrent_kernel" if self.is_rnn else "kernel",
        )

        if self.is_rnn:
            self.layer.cell.recurrent_kernel = kernel
        else:
            self.layer.kernel = kernel

        return self.layer

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return OurWeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)
