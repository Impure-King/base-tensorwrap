import jax.numpy as jnp
from jax import jit, grad, vmap, random
from . import activations, initializers
import tensorwrap

key = random.PRNGKey(0)

class Layer():
    def __init__(self, units = None, trainable = True, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.units = units
        self.train = trainable
    def build(self, input_shape):
        self.build_input_shape = input_shape
        self.built = True
    def call(self, inputs, *args, **kwargs):
        return inputs
    def add_weight(self, shape = None, dtype = jnp.float32, trainable = True, initializer = None):
        self.dtype = dtype
        self.initializer = initializers.get(initializer)
        return jnp.array(random.normal(key, [shape], dtype = self.dtype))

class Dense(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
    
    def build(self):
        self.kernel = self.add_weight(shape = [self.build_input_shape[-1], self.units])
        self.bias = self.add_weight(shape = [self.units])

    def call(self, inputs):
        x = jnp.matmul(inputs, self.kernel) + self.bias
        return self.activation(x)
