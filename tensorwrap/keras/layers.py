import jax.random

from tensorwrap import Module
import jax.numpy as jnp
from random import randint


class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers, to ensure that they are converted in PyTrees."""

    def __init__(self, trainable=True, dtype=None, dynamic=False, **kwargs) -> None:
        super().__init__(self, trainable=trainable, dtype=dtype, dynamic=dynamic, **kwargs)

    def add_weights(self, shape=None, initializer=None, trainable=True):
        if initializer == 'zeros':
            return jnp.zeros(shape, dtype=jnp.float32)

        elif initializer == 'glorot_normal':
            key = jax.random.PRNGKey(randint(1, 10))
            return jax.random.normal(key, (shape,))

        elif initializer == 'glorot_uniform':
            key = jax.random.PRNGKey(randint(1, 5))
            return jax.random.uniform(key, (shape,))

    def __call__(self):
        pass


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
        super().__init__(units,
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

    def build(self, input_shape):
        self.kernel = self.add_weights(shape=(input_shape[-1], self.units),
                                       initializer=self.kernel_initializer)
        self.bias = self.add_weights(shape=(input_shape[-1], self.units),
                                     initializer=self.bias_initializer)

    def __call__(self, *args, **kwargs):