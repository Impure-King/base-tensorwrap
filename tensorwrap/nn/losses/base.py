"""This module aims to provide a workable subclass for all the loss functions."""

from tensorwrap.module import Module

class Loss(Module):

    def __init__(self, name="Loss", *args, **kwargs) -> None:
        super().__init__(name=name)
        pass

    def call(self, y_true, y_pred, *args, **kwargs):
        pass
    