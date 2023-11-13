from tensorwrap.nn.losses import Loss
import jax
import jax.numpy as jnp
import optax

class SparseCategoricalCrossentropy(Loss):
    def __init__(self, from_logits = False) -> None:
        super().__init__()
        self.from_logits = from_logits
    
    @jax.jit
    def call(self, labels, logits):
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean() 