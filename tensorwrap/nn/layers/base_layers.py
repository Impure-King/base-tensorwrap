# Stable Modules:
from typing import Tuple, Optional

from jax.random import PRNGKey
from jaxtyping import Array

# Custom built Modules:

from tensorwrap.module import Module
from tensorwrap.nn.initializers import GlorotNormal, Initializer


# Custom Trainable Layer

class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers, to ensure that they are converted in PyTrees."""

    _name_tracker: int = 1

    def __init__(self, name: Optional[str] = "Layer", *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)

    def add_weights(self, shape: Tuple[int, ...], initializer:Initializer = GlorotNormal()):
        """Useful method inherited from layers.Layer that adds weights that can be trained.
        ---------
        Arguments:
            - shape: Shape of the inputs and the units
            - initializer (Optional): The initial values of the weights. Defaults to tensorwrap.nn.initializers.GlorotNormal()
            - name(Optional): The name of the weight. Defaults to "unnamed weight".
            - trainable (Optional) - Not required or implemented yet. 
        """
        weight:Array = initializer(shape)
        return weight
 

# Inspection Fixes:
Layer.__module__ = "tensorwrap.nn.layers"

