from . import activations, initializers
from jax import numpy as jnp
import jax
from functools import partial
from .models.modules import Module


class Layer(Module):
    def build(self):
        self.built = True

    @property
    def trainable_variables(self):

        return self.kernel, self.bias
    
    def update(self, weights):
        self.kernel = weights[0]
        self.bias = weights[1]


class Linear(Module):
    """A Dense layer is a layer where all the nuerons are fully connected and then transformed, through matrix multiplication."""
    def __init__(self, 
                in_features,
                out_features,
                bias = True,
                activation = None, 
                kernel_initializer = None, bias_initializer = 'zeros', built = False, dynamic = False, input_shape = [1]):
        super().__init__(
            in_features = in_features,
            out_features = out_features,
            bias = bias,
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
        out = jnp.matmul(jnp.transpose(inputs), self.kernel) + self.bias
        # Uncomment and replace once the activations are completed.
        # return self.activation(out)
        return jnp.squeeze(out)
