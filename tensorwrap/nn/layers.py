from . import activations, initializers
from jax import numpy as jnp
import jax
import tensorwrap as tf
from functools import partial
from .models import modules as md


class Layer(md.Module):
    def build(self):
        self.built = True

    @property
    def trainable_variables(self):

        return self.kernel, self.bias
    
    def update(self, weights):
        self.kernel = weights[0]
        self.bias = weights[1]


class Dense(Layer):
    """A Dense layer is a layer where all the nuerons are fully connected and then transformed, through matrix multiplication."""
    def __init__(self, 
                units, 
                activation = None, 
                kernel_initializer = None, bias_initializer = 'zeros', built = False, dynamic = False, input_shape = [1]):
        super().__init__(
            units = units, 
            activation = activation, 
            kernel_initializer = kernel_initializer, 
            bias_initializer = bias_initializer,
            built = built,
            dynamic = dynamic,
            input_shape = input_shape)
        # Uncomment once activation page is started:
        # self.activation = activations.get(activation)
    
    def unit(self):
        return self.units

    def build(self, shape):
        self.kernel = tf.add_weight(self, shape = [shape[-1], self.units])
        self.bias = tf.add_weight(self, initializer = self.bias_initializer, shape = [self.units])
        super().build()
    
    def __call__(self, inputs):
        out = jnp.matmul(inputs, self.kernel) + self.bias
        # Uncomment and replace once the activations are completed.
        # return self.activation(out)
        return jnp.squeeze(out)
