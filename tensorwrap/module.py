from abc import ABCMeta, abstractmethod


class Module(metaclass=ABCMeta):
    """This is the base class for all types of functions and components.
    This is going to be a static type component, in order to allow jit.compile
    from jax and accelerate the training process."""

    @abstractmethod
    def __init__(self, **kwargs):
        self.dict = kwargs

        for keys in dict:
            setattr(self, keys, dict[keys])

    @abstractmethod
    def __call__(self, *args, **kwargs):
        for keys in kwargs:
            setattr(self, keys, kwargs[keys])
