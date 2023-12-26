import jax
import random
import jax.numpy as jnp
from tensorwrap.module import Module

@jax.vmap
def from_tensor_slices(*args):
    return jnp.array([*args])

class Dataloader(Module):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.data_length = self.data.shape[1]
    
    @staticmethod
    def from_tensor_slices(*args):
        return Dataloader(from_tensor_slices(*args))
    
    def batch(self, batch_size, drop_remainder=True, axis=0):
        
        # Validating size:
        if len(self.data) < batch_size:
            raise ValueError("batch_size can't be greater than data size.")
        
        # Finding remainder:
        remainder = self.data.shape[axis]%batch_size

        if drop_remainder:
            batch_data_prep = self.data[0:self.data.shape[axis] - remainder]
        else:
            batch_data_prep = self.data
        
        batched_data = batch_data_prep.reshape((-1, batch_size) + self.data.shape[1:])
        return Dataloader(batched_data)
    
    def map(self, function):
        new_data = jax.numpy.stack([function(i) for i in self.data])
        return Dataloader(new_data)

    def vmap(self, function):
        """The vectorized version of map that works well for most arrays."""
        new_data = jax.vmap(function)(self.data)
        return Dataloader(new_data)
    
    def shuffle(self, key=random.randint(1, 42), axis=0):
        new_data = jax.random.permutation(jax.random.PRNGKey(key), self.data, axis=axis)
        return Dataloader(new_data)
    
    def first(self):
        for tensor in self.data:
            return tensor
    
    def __iter__(self):
        return iter(self.data)
    
    @property
    def shape(self):
        return self.data.shape
    
    def len(self):
        return len(self.data)