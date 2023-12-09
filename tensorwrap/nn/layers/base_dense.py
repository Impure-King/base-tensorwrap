# Stable Modules:
import jax
from typing import Optional

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
        - use_bias (Optional, bool): A boolean signifying whether to include a bias term. Defaults to ``True``.
        - kernel_initializer (Optional, Initializer): An initializer class that initializes kernel weight. Defaults to ``tensorwrap.nn.intializers.GlorotUniform()``.
        - kernel_initializer (Optional, Initializer): An initializer class that initializes bias weight. Defaults to ``tensorwrap.nn.intializers.Zeros()``.
        - activation (Optional, str): A string that indicates the activation function to be applied before returning the output. Defaults to ``None``.
        - name (Optional, str): A string that indicates the name of the layer. Defaults to ``"Dense"``.
        - trainable (Optional, bool): A boolean denoting whether the layer should be trainable or not.
    """


    def __init__(self,
                 units: int,
                 use_bias: Optional[bool] = True,
                 kernel_initializer: Optional[Initializer] = GlorotUniform(),
                 bias_initializer: Optional[Initializer] = Zeros(),
                 activation: Optional[str] = "None",
                 name: Optional[str] = "Dense",
                 *args,
                 **kwargs):
        super().__init__(name=name,
                         *args,
                         **kwargs)
        self.units:int = units
        self.use_bias:bool = use_bias
        self.kernel_initializer:Initializer = kernel_initializer
        self.bias_initializer:Initializer = bias_initializer
        self.activation:function = Activation.get_activation(activation)

        # Checking each argument:
        argument_list = [("units", units, int),
                         ("use_bias", use_bias, bool),
                         ("kernel_initializer", kernel_initializer, Initializer),
                         ("bias_initializer", bias_initializer, Initializer),
                         ("activation", activation, str)]
        
        for arg_name, arg_val, type in argument_list:
            if not isinstance(arg_val, type):
                raise TypeError(f"""Raised from {self.name}.
                                Argument ``{arg_name}`` is not type ``{type}``.
                                Current argument type: {type(arg_val)}""")

    def build(self, inputs: jax.Array):
        """A build method specifies how the parameters shall be assigned to ``self.params``.
        It allows for the ``init`` method to build the parameters and assign the overall parameters.
        ---------
        Arguments:
            - inputs (Array): An array who's shapes can be determined for parameter building.
        """
        super().build()
        input_shape = jax.numpy.shape(inputs)

        if len(input_shape) < 2:
            raise ValueError(f"Input dimensions is {len(input_shape)}. Expected Input Dimensions of 2."
                             "Use `tensorwrap.nn.expand_dims(inputs, axis=1)` to increase input shape.")

        self.kernel = self.add_weights(shape=(input_shape[-1], self.units),
                                       initializer=self.kernel_initializer)
        
        if self.use_bias:
            self.bias = self.add_weights(shape=(self.units,),
                                         initializer=self.bias_initializer)
            
    def call(self, inputs: jax.Array) -> jax.Array:
        if not self.use_bias:
            x = inputs @ self.kernel
        else:
            x = inputs @ self.kernel + self.bias
        return self.activation(x)

# PyTorch Implementation:

class Linear(Layer):
    def __init__(self,
                 in_shape: int,
                 out_shape: int,
                 use_bias: bool = True,
                 kernel_initializer: Initializer = GlorotUniform(),
                 bias_initializer: Initializer = Zeros(),
                 name: str = "Linear",
                 *args,
                 **kwargs) -> None:
        super().__init__(name=name,
                         *args,
                         **kwargs)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
    
    def build(self, inputs):
        super().build()
        input_shape = self.in_shape
        self.kernel = self.add_weights(shape = (input_shape, self.out_shape),
                                       initializer = self.kernel_initializer)
        if self.use_bias:
            self.bias = self.add_weights(shape = (self.out_shape,),
                                         initializer = self.bias_initializer)


    def call(self, inputs: jax.Array) -> jax.Array:
        if not self.use_bias:
            return inputs @ self.kernel
        
        x = inputs @ self.kernel + self.bias
        return x

Dense.__module__ = "tensorwrap.nn.layers"
Linear.__module__ = "tensorwrap.nn.layers"