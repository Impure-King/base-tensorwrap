import jax
import random
import tensorwrap as tf
from tensorwrap.module import Module


class Dataloader(Module):
    def __init__(self, data) -> None:
        self.data = jax.numpy.array(data)
    
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
    

class Dataset(Module):
    def __init__(self) -> None:
        pass
    
    @classmethod
    def from_tensor_slices(self, data_slice) -> None:
        if not isinstance(data_slice, tuple) and not isinstance(data_slice, list):
            raise ValueError("Tensor slices must be a list or tuple for efficient processing.")

        # Getting the length of the array.
        self._no_arrays = len(data_slice)

        for i in range(self._no_arrays):
            setattr(self, str(i + 1), Dataloader(data_slice[i]))
        
        
        
    def batch(self, batch_size, drop_remainder = True, axis=0):
        for i in range(self._no_arrays):
            setattr(self, str(i + 1), self.__getattribute__(str(i + 1)).batch(batch_size, drop_remainder, axis))
    
    def map(self, function):
        for i in range(self._no_arrays):
            setattr(self, str(i + 1), self.__getattribute__(str(i + 1)).map(function))
    
    def shuffle(self, key = random.randint(1, 42), axis=0):
        for i in range(self._no_arrays):
            setattr(self, str(i + 1), self.__getattribute__(str(i + 1)).batch(key, axis))
    

    def __iter__(self):
        self.full_data = []
        
        for i in range(self._no_arrays):
            self.full_data.append(self.__getattribute__(str(i+1)))
        
        return zip(*self.full_data)