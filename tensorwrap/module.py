import jax
from typing import Any
from abc import ABCMeta, abstractmethod
from jax import jit
from jax.tree_util import register_pytree_node_class
from inspect import signature
import tensorwrap as tf

# All classes allowed for export.
__all__ = ["Module"]


@register_pytree_node_class
class Module(metaclass=ABCMeta):
    """ Basic neural network module class.
    
    A module class is a named container that acts to transform all subclassed containers into
    pytrees by appropriately defining the tree_flatten and tree_unflatten. Additionally, it 
    defines the trackable trainable_variables for all the subclasses.
    
    NOTE: Due to the limited functionality of the Module class, it isn't recommended for 
    general or even research use. Additionally, the Module class lacks functionality of the 
    original TensorFlow implementation, so avoid any implementations of this class. Only made
    for internal use and public api placeholder."""

    def __init__(self) -> None:
        """Helps instantiate the class and assign a self.trainable_variables to subclass."""
        self.trainable_variables = {}
        self.__unflattened = False

    @classmethod
    def __init_initialize__(cls):
        """An extremely dangerous method which empties our the __init__ method and then create an instance. 
        After, repurposing the __init__ again and it returns an instance with an empty init function. DO NOT USE for external uses."""
        
        # function to replace __init__ temporarily:
        def init_rep(self):
            self.trainable_variables = {}
        prev_init = cls.__init__ # Storing __init__ functionality
        
        # Emptying and creating a new instance
        cls.__init__ = init_rep
        instance = cls()

        # Reverting changes and returning instance
        cls.__init__ = prev_init

        return instance

    def __init_subclass__(cls) -> None:
        """Used to convert and register all the subclasses into Pytrees."""
        register_pytree_node_class(cls)

    def __call__(self, *args, **kwargs) -> Any:
        """Maintains the call() convention for all subclasses."""
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """Acts as an abstract method to force all implementation to occur in the `call` method."""
        pass

    def tree_flatten(self):
        self.__unflattened = True
        leaves = {}
        for key in self.trainable_variables:
            leaves[key] = self.trainable_variables[key]
        
        # Removing trainable_variables:
        aux_data = vars(self).copy()

        aux_data.pop("trainable_variables")
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        instance = cls.__init_initialize__()
        vars(instance).update(children)
        vars(instance).update(aux_data)
        return instance
    




    # Broken/Unnecessary methods and features:

    # def __init__(self, *args, **kwargs) -> None:
    #     # Setting all the argument attributes:
    #     for key, value in enumerate(args):
    #         setattr(self, f"arg_{key}", value)

    #     # Setting all the keyword argument attributes:
    #     for key, value in kwargs.items():
    #         setattr(self, key, value)


