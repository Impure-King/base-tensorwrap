import jax.numpy as jnp
import jax
import random

key = jax.random.PRNGKey(random.randint(1, 10))

def add_weight(self, shape, dtype = jnp.float32, trainable = True, initializer = None):
    self.dtype = dtype
    # Uncomment once the initializer is made:
    # self.initializer = initializers.get(initializer)
    if initializer == None:
        return jnp.array(jax.random.normal(key, (shape), dtype = self.dtype))
    elif initializer == 'zeros':
        return jnp.array(jnp.zeros((shape), dtype = self.dtype))