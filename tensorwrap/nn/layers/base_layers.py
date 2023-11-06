# Stable Modules:
from random import randint
from typing import Tuple, final

import jax
from jax.random import PRNGKey
from jaxtyping import Array

# Custom built Modules:

from tensorwrap.module import Module
from tensorwrap.nn.initializers import GlorotNormal, GlorotUniform, Initializer, Zeros


# Custom Trainable Layer

class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers, to ensure that they are converted in PyTrees."""

    _name_tracker: int = 1

    def __init__(self, name: str = "Layer") -> None:
        super().__init__(name=name)

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
            self.params[self.name][name] = weight

        return weight

    # Future idea to automize layer building.
    # def compute_output_shape(self):
    #     raise NotImplementedError("Method `compute_output_shape` has not been implemented.")

    def get_weights(self, name):
        return self._params[name]

    # @final
    # def __call__(self, params: dict, inputs: Array, *args, **kwargs):
    #     if not self.built:
    #         self.init_params(inputs)
    #         params = self.get_trainable_params() # To be accepted in the format.
    #     out = self.call(params[self.name], inputs, *args, **kwargs)
    #     return out
    
    def call(self, params, inputs):
        if NotImplemented:
            raise NotImplementedError("Call Method Missing:\nPlease define the control flow in the call method.")




    


# Inspection Fixes:
Layer.__module__ = "tensorwrap.nn.layers"

