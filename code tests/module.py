import jax
from jax import numpy as jnp
from jax.tree_util import tree_structure
from typing import NamedTuple


# Library imports:

from tensorwrap import (Module,
                        nn)


# Testing the main module class for Pytree unrolling:
class Container(Module):
    def __init__(self, weight:int) -> None:
        super().__init__()
        self.weights = weight

model = Container(1)

model2 = NamedTuple("model2", weights=int)(1)

print(tree_structure(model))
print(tree_structure(model2))



# Testing for jit compilation:
@jax.jit
def models(model):
    return model

print(models(model))
print(models(model2))