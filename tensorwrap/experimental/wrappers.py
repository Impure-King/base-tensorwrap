import jax
from jax import numpy as jnp


# class HashableArrayWrapper:
#     def __init__(self, val, hash_function):
#         self.var = val
#         self.function = hash_function
#
#     def __hash__(self):
#         return self.function(self.var)
#
#     def __eq__(self, other):
#         return (isinstance(other, HashableArrayWrapper) and
#                 jnp.all(jnp.equal(self.val, other.val)))


class MemoryErrors(Exception):
    """Exception raised for various memory errors.
       Arguments:
          - Message to display when catching the error. """
    def __init__(self, error=None) -> None:
        self.err = error
        super().__init__(self.err)
