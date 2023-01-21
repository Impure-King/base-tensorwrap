from abc import ABCMeta, abstractmethod
from jax.tree_util import register_pytree_node_class


class Module_Base(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        for keys in args:
            setattr()
@register_pytree_node_class
class Module(metaclass=ABCMeta):
    """This is the base class for all types of functions and components.
    This is going to be a static type component, in order to allow jit.compile
    from jax and accelerate the training process."""

    @abstractmethod
    def __init__(self, **kwargs):
        for keys in dict:
            setattr(self, keys, dict[keys])

    @abstractmethod
    def __call__(self, *args, **kwargs):
        for keys in kwargs:
            setattr(self, keys, kwargs[keys])

    def tree_flatten(self):
        tuple = []
        for key in self.dict:
            tuple.append(dict[])
        return ((x, y), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
