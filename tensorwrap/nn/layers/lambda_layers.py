# Stable Modules
from typing import Any

import jax.numpy as jnp
from jax import jit

# Custom build modules
from tensorwrap.module import Module

__all__ = ["Lambda", "Flatten"]

class Lambda(Module):
    """A superclass for layers without trainable variables."""
    _name_tracker = 0
    def __init__(self, training_mode=False, name="Lambda") -> None:
        self.training_mode = training_mode
        self.name = f"{name}:{Lambda._name_tracker}"

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.__call__ = cls.call
    
    def set_training_mode(self, training_mode):
        self.training_mode = training_mode
        print(f"Model Trainable Mode: {self.training_mode}")

    def __call__(self, *args, **kwargs) -> Any:
        pass

    def __repr__(self) -> str:
        return f"<tf.{self.name}>"

class Flatten(Lambda):
    def __init__(self, input_shape = None, name="Flatten") -> None:
        super().__init__()
        if input_shape is None:
            self.input_shape = -1
        else:
            self.input_shape = jnp.prod(jnp.array(input_shape))

    
    def call(self, params, inputs) -> Any:
        return jnp.reshape(inputs, [inputs.shape[0], self.input_shape])

# Inspection Fixes:
Lambda.__module__ = "tensorwrap.nn.layers"
Flatten.__module__ = "tensorwrap.nn.layers"