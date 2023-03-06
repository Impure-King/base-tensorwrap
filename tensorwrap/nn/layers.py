import jax.random
from functools import partial
from jax import jit
from jaxtyping import Array
from tensorwrap.module import Module
import tensorwrap as tf
import jax.numpy as jnp
from random import randint

# Custom Layer

class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers, to ensure that they are converted in PyTrees."""

    def __init__(self, trainable=True, dtype=None, **kwargs) -> None:
        self.trainable = trainable
        self.dtype = dtype
        self.kwargs = kwargs
        self.trainable_variables = []

    @classmethod
    def add_weights(self, shape=None, initializer='glorot_uniform', trainable=True, name=None):
        """Useful method inherited from layers.Layer that adds weights that can be trained.
        ---------
        Arguments:
            - shape: Shape of the inputs and the units
            - initializer: The initial values of the weights
            - trainable - Not required or implemented yet."""

        if initializer == 'zeros':
            return jnp.zeros(shape, dtype=jnp.float32)

        elif initializer == 'glorot_normal':
            key = jax.random.PRNGKey(randint(1, 10))
            return jax.random.normal(key, shape, dtype = tf.float32)

        elif initializer == 'glorot_uniform':
            key = jax.random.PRNGKey(randint(1, 10))
            return jax.random.uniform(key, shape, dtype = tf.float32)


    def call(self) -> None:
        # Must be defined to satisfy arbitrary method.
        pass
    

    def __call__(self, inputs):
        # This is to compile, in not built.
        if not self.built:
            self.build(tf.shape(inputs))
        out = self.call(inputs)
        return out

    # Previous build is depracated.
    # def build(self, input_shape, input_check: bool = True):
    #     input_dims = len(input_shape)
    #     if input_dims <= 1 and input_check:
    #         print("Input to the Dense layer has dimensions less than 1. \n"
    #               "Use tf.expand_dims or tf.reshape(-1, 1) in order to expand dimensions.")
    #     self.trainable_variables = [self.kernel, self.bias]
    #     self.built = True


# Dense Layer:

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
        self.built = False
        super(Module, self).__init__(units=units,
                                     activation=activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     activity_regularizer=activity_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     bias_constraint=bias_constraint,
                                     dynamic=not tf.test.is_device_available())

    def build(self, input_shape:int):
        self.kernel = self.add_weights(shape=(input_shape, self.units),
                                       initializer=self.kernel_initializer,
                                       name="kernel")
        self.bias = self.add_weights(shape=(self.units),
                                     initializer=self.bias_initializer,
                                     name="bias")
        self.trainable_variables = [self.kernel, self.bias]
        self.built = True

    def call(self, inputs: Array) -> Array:
        if self.use_bias == True:
            return jnp.matmul(inputs, self.trainable_variables[0]) + self.trainable_variables[1]
        else:
            return jnp.matmul(inputs, self.trainable_variables[0], inputs)


