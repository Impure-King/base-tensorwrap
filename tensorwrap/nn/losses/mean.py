from tensorwrap.nn.losses import Loss
from jax import numpy as jnp
from jax import jit

__all__ = ["mse", "mae"]

class MeanSquaredError(Loss):
    def __init__(self, name="MeanSquaredError", *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)
        pass

    
    def call(self, y_true, y_pred):
        return jnp.mean((y_true - y_pred)**2)

class MeanAbsoluteError(Loss):
    def __init__(self, name="MeanAbsoluteError", *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)
        pass

    
    def call(self, y_true, y_pred):
        return jnp.mean(jnp.abs(y_true - y_pred))

# Inspection Fixes:
MeanSquaredError.__module__ = "tensowrap.nn.losses"
MeanAbsoluteError.__module__ = "tensorwrap.nn.losses"


# Adding proper names:
MeanSquaredError.__repr__ = "<function mse>"
MeanAbsoluteError.__repr__ = "<function mae>"