import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from tensorwrap.nn import activation
from tensorwrap.nn import initializers
from tensorwrap.nn import regularizers
from tensorwrap.nn import constraints
from tensorwrap import self
seed = random.PRNGKey(0)

class Layers:
    def __init__(self):
        pass

class Dense:
    def __init__(self, 
                units,
                activation_fn = None, 
                use_bias = True, 
                kernel_initializer = 'glorot_uniform',
                bias_initializer = 'zeros',
                kernel_regularizer = None,
                bias_regularizer = None,
                activity_regularizer = None,
                kernel_constraint = None,
                bias_constraint = None,
                **kwargs):
        super(Dense, self).__init__(activity_regularizer = activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f"Received invalid input for 'units', expected "
                             f"a positive integer, but instead got {units}.")
        
        self.activation = activation.get(activation_fn)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape = [input_shape[-1], self.units], 
            name = 'kernel', 
            dtype = jnp.float32,
            initializers = self.kernel_initializer, 
            trainable = True)
        
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape = [self.units],
                initializers = self.bias_initializer,
                dtype = jnp.float32,
                trainable = True,
            )
        
        super().build(input_shape)
    
    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)