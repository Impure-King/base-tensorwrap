import jax.random
from jax import jit
from jaxtyping import Array
from tensorwrap.module import Module
import jax.numpy as jnp
from random import randint
from tensorwrap.test import is_gpu_available


class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers, to ensure that they are converted in PyTrees."""

    # Removing initializer due to a bug:
    def __init__(self, trainable=True, dtype=None, dynamic=not is_gpu_available(), **kwargs) -> None:
        self.trainable = trainable
        self.dtype = dtype
        self.dynamic = dynamic
        self.kwargs = kwargs
        self.built = False

    def add_weights(self, shape=None, initializer='glorot_uniform', trainable=True):
        """Useful method inherited from layers.Layer that adds weights that can be trained.
        Arguments:
            - shape: Shape of the inputs and the units
            - initializer: The initial values of the weights
            - trainable - Not required or implemented yet."""

        if initializer == 'zeros':
            return jnp.zeros(shape, dtype=jnp.float32)

        elif initializer == 'glorot_normal':
            key = jax.random.PRNGKey(randint(1, 10))
            return jax.random.normal(key, shape)

        elif initializer == 'glorot_uniform':
            key = jax.random.PRNGKey(randint(1, 5))
            return jax.random.uniform(key, shape)

    def build(self, input_shape):
        input_dims = len(input_shape)
        if input_dims <= 1:
            raise ValueError("Input to the Dense layer has dimensions less than 1."
                             "Use tf.expand_dims or tf.reshape(-1, 1) in order to expand dimensions.")
        self.built = True

    def call(self):
        # Must be defined to satisfy arbitrary method.
        pass

    def __call__(self, inputs):
        # This will be called when called normally.
        if not self.built:
            self.build(inputs.shape)
        if not self.dynamic:
            function = jit(self.call)
        else:
            function = self.call
        # Ensures a tensorflow-like API by calling the call function inside __call__.
        out = function(inputs)
        return out


class Dense(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 *args,
                 **kwargs):
        super(Module, self).__init__(units=units,
                                     activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     activity_regularizer=activity_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     bias_constraint=bias_constraint)
        self.built = False

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = self.add_weights(shape=(input_shape[-1], self.units),
                                       initializer=self.kernel_initializer)
        self.bias = self.add_weights(shape=(self.units),
                                     initializer=self.bias_initializer)

    def call(self, inputs: Array) -> Array:
        if self.use_bias == True:
            return jnp.matmul(self.kernel, inputs) + self.bias
        else:
            return jnp.matmul(self.kernel, inputs)

