from tensorwrap.nn.metrics import Metric
import jax.numpy as jnp
from jax import jit

class Accuracy(Metric):
    def __init__(self, one_hot = True, name: str | None = "Module"):
        super().__init__(name)
        self.one_hot = one_hot

    @jit
    def call(self, y_true, y_pred, *args, **kwargs):
        """Computes the accuracy metric.

        Args:
            y_true (jax.numpy.ndarray): The true labels with shape (batch_size,).
            y_pred (jax.numpy.ndarray): The predicted logits or class probabilities with shape (batch_size, num_classes).
            from_logits (bool, optional): Whether the predicted values are logits or class probabilities.
                Defaults to True.

        Returns:
            float: The accuracy value.
        """
        if self.one_hot:
            y_pred = jnp.argmax(y_pred, axis=1)
        return jnp.mean(y_pred == y_true) * 100