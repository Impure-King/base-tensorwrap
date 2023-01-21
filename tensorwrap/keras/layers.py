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

    def __init__(self, trainable=True, dtype=None, dynamic=False, **kwargs) -> None:
        super().__init__(self, trainable=trainable, dtype=dtype, dynamic=dynamic, **kwargs)

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
        pass

    def __call__(self, inputs, static=is_gpu_available()):
        if not self.built:
            self.build(inputs.shape)
        if static:
            function = jit(self.call)
        else:
            function = self.call
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
                 **kwargs):
        super().__init__(units=units,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
        self.built = False

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = self.add_weights(shape=(input_shape[-1], self.units),
                                       initializer=self.kernel_initializer)
        self.bias = self.add_weights(shape=(self.units),
                                     initializer=self.bias_initializer)

    def call(self, inputs: Array) -> Array:
        return jnp.matmul(self.kernel, inputs) + self.bias

