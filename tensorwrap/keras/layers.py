import jax.random
from jax import jit
from jaxtyping import Array
from tensorwrap.module import Module
import tensorwrap as tf
import jax.numpy as jnp
from random import randint


class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers, to ensure that they are converted in PyTrees."""

    def __init__(self, trainable=True, dtype=None, dynamic=False, **kwargs) -> None:
        self.trainable = trainable
        self.dtype = dtype
        self.dynamic = dynamic
        self.kwargs = kwargs

    def add_weights(self, shape=None, initializer='glorot_uniform', trainable=True, name=None):
        """Useful method inherited from layers.Layer that adds weights that can be trained.
        Arguments:
            - shape: Shape of the inputs and the units
            - initializer: The initial values of the weights
            - trainable - Not required or implemented yet."""

        if initializer == 'zeros':
            return name, jnp.zeros(shape, dtype=jnp.float32), trainable

        elif initializer == 'glorot_normal':
            key = jax.random.PRNGKey(randint(1, 10))
            return name, jax.random.normal(key, shape), trainable

        elif initializer == 'glorot_uniform':
            key = jax.random.PRNGKey(randint(1, 5))
            return name, jax.random.uniform(key, shape), trainable

    def build(self, input_shape):
        input_dims = len(input_shape)
        if input_dims <= 1:
            raise ValueError("Input to the Dense layer has dimensions less than 1."
                             "Use tf.expand_dims or tf.reshape(-1, 1) in order to expand dimensions.")
        self.built = True

    def call(self) -> None:
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
        self.built = False
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
                                     bias_constraint=bias_constraint,
                                     built=False,
                                     dynamic=not tf.test.is_gpu_available(),
                                     weights=[],
                                     trainable_weights=[],
                                     trainable_variables=[])

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = self.add_weights(shape=(input_shape[-1], self.units),
                                       initializer=self.kernel_initializer,
                                       name="kernel")
        self.bias = self.add_weights(shape=(self.units),
                                     initializer=self.bias_initializer,
                                     name="bias")

        # Used to return the weights with trainable status
        self.weights = [self.kernel, self.bias]
        # Used to return trainable weights with their name:
        self.trainable_weights = [x[:2] for x in self.weights if x[2]]
        # Used to return trainable variables that can be used to apply gradients.
        self.trainable_variables = [x[1] for x in self.trainable_weights]

    def call(self, inputs: Array) -> Array:

        if self.use_bias == True:
            return jnp.matmul(inputs, self.kernel[1]) + self.bias[1]
        else:
            return jnp.matmul(inputs, self.kernel[1])
