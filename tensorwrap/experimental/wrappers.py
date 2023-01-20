import jax
from jax import numpy as jnp


class HashableArrayWrapper:
    def __init__(self, val, hash_function):
        self.var = val
        self.function = hash_function

    def __hash__(self):
        return self.function(self.var)

    def __eq__(self, other):
        return (isinstance(other, HashableArrayWrapper) and
                jnp.all(jnp.equal(self.val, other.val)))
