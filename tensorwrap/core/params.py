# Stable Modules:
import jax
import jax.numpy as jnp
from typing import Any, Dict

# Custom Modules:
from tensorwrap import Module


# Creating a parameter class object:
class Params(Module):
    def __init__(self, name: str):
        super().__init__(name)
        self._values = {self.name: {}}
    
    def set_complete_params(self, params):
        self._values = params

    def set_value(self, name: str, value: Any):
        self._values[self.name][name] = value

    def get_value(self, name):
        return name