from typing import Any, Optional
from tensorwrap.module import Module
import jax
import jax.numpy as jnp

class Metric(Module):
    def __init__(self, name: str | None = "Module"):
        super().__init__(name)
        self.state = []
    
    def reset(self):
        self.state = []

    def __call__(self, y_true, y_pred, *args, **kwargs):
        x = self.call(y_true, y_pred, *args, **kwargs)
        self.state.append(x)
        return jnp.mean(jnp.array(self.state))
