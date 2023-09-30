import jax
import random

import tensorwrap as tw
from tensorwrap.nn.layers import Lambda


class Dropout(Lambda):
    def __init__(self, rate, shape = None, seed = None, training_mode = False, name="Lambda") -> None:
        super().__init__(training_mode=training_mode, 
                         name=name)
        self.rate = rate
        self.input_shape = shape
        self.seed = random.randint(1, 5) if seed is None else seed
    
    @staticmethod
    def dropout(inputs, rate, noise_shape=None, seed=None):
        seed = jax.random.PRNGKey(seed)
        keep_prob = 1.0 - rate
        # The `noise_shape` may contain `None` so we need to convert it
        # into a concrete shape before passing it on to jax.
        noise_shape = inputs.shape
        mask = jax.random.bernoulli(seed, p=keep_prob, shape=noise_shape)
        mask = jax.numpy.broadcast_to(mask, inputs.shape)
        return jax.lax.select(
            mask, inputs / keep_prob, jax.numpy.zeros_like(inputs)
        )

    @jax.jit
    def call(self, params, inputs):
        if self.training_mode and self.rate > 0:
            self.seed = random.randint(1, 36)
            return self.dropout(
                inputs,
                self.rate,
                self.input_shape if self.input_shape is None else inputs.shape,
                self.seed
            )
        return inputs

# Inspection Fixes:
Dropout.__module__ = "tensorwrap.nn.layers"