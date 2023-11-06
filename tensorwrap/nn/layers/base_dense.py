# Stable Modules:
import jax
from jaxtyping import Array

# Custom built Modules:
from tensorwrap.nn.activations import Activation

from tensorwrap.nn.initializers import GlorotUniform, Initializer, Zeros
from tensorwrap.nn.layers import Layer

# TensorFlow Implementation:

class Dense(Layer):
    """ A fully connected layer that applies linear transformation to the inputs.
    ---------
    Arguments:
        - units (int): A positive integer representing the output shape.
        - use_bias (Optional, bool): A boolean signifying whether to include a bias term.
        - kernel_initializer (Optional, Initializer): An initializer class that initializes kernel weight. Defaults to tensorwrap.nn.intializers.GlorotUniform().
        - kernel_initializer (Optional, Initializer): An initializer class that initializes bias weight. Defaults to tensorwrap.nn.intializers.Zeros().
    """


    def __init__(self,
                 units: int,
                 use_bias: bool = True,
                 kernel_initializer: Initializer = GlorotUniform(),
                 bias_initializer: Initializer = Zeros(),
                 activation: str = "None",
                 name: str = "Dense"):
        super().__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = Activation.get_activation(activation)

    def build(self, inputs):
        super().build()
        input_shape = jax.numpy.shape(inputs)

        if len(input_shape) < 2:
            raise ValueError(f"Input dimensions is {len(input_shape)}. Expected Input Dimensions of 2."
                             "Use `tensorwrap.nn.expand_dims(inputs, axis=1)` to increase input shape.")

        self.kernel = self.add_weights(shape=(input_shape[-1], self.units),
                                       initializer=self.kernel_initializer,
                                       name="kernel")
        if self.use_bias:
            self.bias = self.add_weights(shape=(self.units,),
                                         initializer=self.bias_initializer,
                                         name="bias")

    def call(self, params: dict, inputs: Array) -> Array:
        if not self.use_bias:
            x = inputs @ params['kernel']
        else:
            x = inputs @ params['kernel'] + params['bias']
        return self.activation(x)

# PyTorch Implementation:

class Linear(Layer):
    def __init__(self,
                 in_shape: int,
                 out_shape: int,
                 use_bias: bool = True,
                 kernel_initializer: Initializer = GlorotUniform(),
                 bias_initializer: Initializer = Zeros(),
                 training_mode=False,
                 name: str = "Layer",
                 *args,
                 **kwargs) -> None:
        super().__init__(training_mode=training_mode,
                         name=name,
                         *args,
                         **kwargs)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.init_params(None)
    
    def build(self, inputs):
        input_shape = self.in_shape
        self.kernel = self.add_weights(shape = (input_shape, self.out_shape),
                                       initializer = self.kernel_initializer,
                                       name = "kernel")
        if self.use_bias:
            self.bias = self.add_weights(shape = (self.out_shape,),
                                         initializer = self.bias_initializer,
                                         name="bias")


    @jax.jit
    def call(self, params: dict, inputs: Array) -> Array:
        if not self.use_bias:
            return inputs @ params['kernel']
        
        x = inputs @ params['kernel'] + params['bias']
        return x

Dense.__module__ = "tensorwrap.nn.layers"
Linear.__module__ = "tensorwrap.nn.layers"