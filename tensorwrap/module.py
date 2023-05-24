import jax
from typing import Any
from abc import ABCMeta, abstractmethod
from jax import jit
from jax.tree_util import register_pytree_node_class

# All classes allowed for export.
__all__ = ["Module"]


class Module(metaclass=ABCMeta):
    """ This is the most basic template that defines all subclass items to be a pytree and accept arguments flexibly.
    Don't use this template and instead refer to the Module Template, in order to create custom parts. If really needed,
    use the PyTorch variation which will be suited for research."""

    def __init__(self, *args, **kwargs) -> None:
        # Setting all the argument attributes:
        for key, value in enumerate(args):
            setattr(self, f"arg_{key}", value)

        # Setting all the keyword argument attributes:
        for key, value in kwargs.items():
            setattr(self, key, value)

    # This function is responsible for making the subclasses into PyTrees:
    def __init_subclass__(cls) -> None:
        register_pytree_node_class(cls)

    def __call__(self, *args, **kwargs) -> Any:
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, *args, **kwargs):
        pass

    def tree_flatten(self):
        dic = vars(self).copy()
        aux_data = {}
        leaves = []
        # Removes the dynamic elements:
        for key in vars(self).keys():
            if isinstance(dic[key], (str, int, bool)):
                aux_data[key] = dic.pop(key)
            elif isinstance(dic[key], jax.Array):
                leaves.append(dic.pop(key))
        
        leaves = (leaves)
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        instance = cls(*children, **aux_data)
        return instance

