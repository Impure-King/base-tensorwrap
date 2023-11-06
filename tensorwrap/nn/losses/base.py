"""This module aims to provide a workable subclass for all the loss functions."""

from tensorwrap.module import Module

class Loss(Module):

    def __init__(self) -> None:
        super().__init__()
        pass

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.__call__ = cls.call

    def __call__(self, y_true, y_pred, *args, **kwargs):
        raise NotImplementedError("``self.call`` is not defined.")
    