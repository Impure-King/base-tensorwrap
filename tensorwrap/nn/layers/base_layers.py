# Stable Modules:
from random import randint
from typing import Tuple, final

import jax
from jax.random import PRNGKey
from jaxtyping import Array

# Custom built Modules:
# import tensorwrap as tf

from tensorwrap.module import Module
from tensorwrap.nn.initializers import GlorotNormal, GlorotUniform, Initializer, Zeros

__all__ = ["Layer", "Dense"]

# Custom Trainable Layer


class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers, to ensure that they are converted in PyTrees."""

    _name_tracker: int = 1

    def __init__(self, name: str = "Layer") -> None:
        self.built = False
        # Name Handling:
        self.name = name + ":" + str(Layer._name_tracker)
        self.id = Layer._name_tracker
        Layer._name_tracker += 1

        self.trainable_variables = {self.name:{}}

    def add_weights(self, shape: Tuple[int, ...], key = PRNGKey(randint(1, 1000)), initializer:Initializer = GlorotNormal(), name = 'unnamed weight', trainable=True):
        """Useful method inherited from layers.Layer that adds weights that can be trained.
        ---------
        Arguments:
            - shape: Shape of the inputs and the units
            - initializer (Optional): The initial values of the weights. Defaults to tensorwrap.nn.initializers.GlorotNormal()
            - name(Optional): The name of the weight. Defaults to "unnamed weight".
            - trainable (Optional) - Not required or implemented yet. 
        """
        
        weight = initializer(shape)

        # Adding to the trainable variables:
        if trainable:
            self.trainable_variables[self.name][name] = weight

        return weight

    def build(self, inputs):
        pass

    def init_params(self, inputs):
        self.build(inputs)
        self.built=True
        return self.trainable_variables

    # Future idea to automize layer building.
    # def compute_output_shape(self):
    #     raise NotImplementedError("Method `compute_output_shape` has not been implemented.")

    def get_weights(self, name):
        return self.trainable_variables[self.name][name]

    @final
    def __call__(self, params: dict, inputs: Array, *args, **kwargs):
        if not self.built:
            self.init_params(inputs)
            params = self.trainable_variables # To be accepted in the format.
        out = self.call(params[self.name], inputs, *args, **kwargs)
        return out
    
    def call(self, params, inputs):
        if NotImplemented:
            raise NotImplementedError("Call Method Missing:\nPlease define the control flow in the call method.")


    # Displaying the names:
    def __repr__(self) -> str:
        return f"<tf.{self.name}>"


# Dense Layer:

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
                 bias_initializer: Initializer = Zeros()):
        super().__init__()
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, inputs):
        input_shape = jax.numpy.shape(inputs)
        if len(input_shape) < 2:
            raise ValueError(f"Input dimensions is {len(input_shape)}. Expected Input Dimensions of 2.\n Use `tensorwrap.nn.expand_dims(inputs, axis=1)` to increase input shape.")
        self.kernel = self.add_weights(shape = (input_shape[-1], self.units),
                                       initializer = self.kernel_initializer,
                                       name = "kernel")
        if self.use_bias:
            self.bias = self.add_weights(shape = (self.units,),
                                         initializer = self.bias_initializer,
                                         name="bias")
        else:
            self.bias = None
            self.trainable_variables['bias'] = self.bias


    @jax.jit
    def call(self, params: dict, inputs: Array) -> Array:
        # super().call()
        if not self.use_bias:
            return inputs @ params['kernel']
        
        x = inputs @ params['kernel'] + params['bias']
        return x
    


# Inspection Fixes:
Layer.__module__ = "tensorwrap.nn.layers"
Dense.__module__ = "tensorwrap.nn.layers"
