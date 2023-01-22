from typing import Any
from abc import ABCMeta, abstractmethod
from jax.tree_util import register_pytree_node_class


class BaseModule(metaclass=ABCMeta):
    """ This is the most basic template that defines all subclass items to be a pytree and accept arguments flexibly.
    Don't use this template and instead refer to the Module Template, in order to create custom parts. If really needed,
    use the PyTorch variation which will be suited for research."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Setting all the argument attributes:
        for keys in range(len(args)):
            setattr(self, str(args[keys]), args[keys])

        # Setting all the keyword argument attributes:
        for keys in kwargs:
            setattr(self, keys, kwargs[keys])

    # This function is responsible for making the subclasses into PyTrees:
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__()
        register_pytree_node_class(cls)

    @abstractmethod
    def call(self, *args, **kwargs):
        pass



# Creating abstract class:
class Module(BaseModule):
    """This is the base class for all types of functions and components.
    This is going to be a static type component, in order to allow jit.compile
    from jax and accelerate the training process."""

    def tree_flatten(self):
        dic = vars(self).copy()
        try:
            aux_data = vars(self).pop('kwargs')
        except:
            aux_data = None
        children = vars(self).values()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def call(self):
        pass


