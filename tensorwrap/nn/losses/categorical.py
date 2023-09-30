from tensorwrap.nn.losses import Loss
import jax
import optax

class SparseCategoricalCrossentropy(Loss):
    def __init__(self, from_logits = False) -> None:
        super().__init__()
        self.from_logits = from_logits
    @jax.jit
    def call(self, labels, logits):
        num_classes = logits.shape[1]
        labels = jax.nn.one_hot(labels, num_classes)

        if self.from_logits:
            logits = jax.nn.log_softmax(logits)

        loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
        return jax.numpy.mean(loss)
