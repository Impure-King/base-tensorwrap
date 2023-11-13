# Stable Modules:
import jax
from jax import numpy as jnp
from jaxtyping import Array
from termcolor import colored
from typing import Any

# Custom built Modules:
import tensorwrap as tw
from tensorwrap.module import Module
from tensorwrap.nn.layers.base_layers import Layer
from tensorwrap.nn.losses.base import Loss

__all__ = ["Model", "Sequential"]


class Model(Module):
    """A Module subclass that can be further subclasses for more complex models that don't
    work with Sequential class.
    ---------
    Arguments:
        - name (string): The name of the model.
    
    Returns:
        - Model: An empty Model class that has prebuilt methods like predict, evaluate, and can be used with train_states or subclassed in other models.
    
    NOTE: Recommended only for subclassing use.
    """

    def __init__(self, name: str = "Model") -> None:
        super().__init__(name=name) # Loads Module configurations
    
    def predict(self, inputs: jax.Array) -> jax.Array:
        """Returns the predictions, when given inputs for the model.
        
        Arguments:
            - inputs: Proprocessed JAX arrays that can be used to calculate an output."""
        return self.__call__(self.params, inputs)
    
    def evaluate(self, inputs: jax.Array, labels: jax.Array, loss_fn: Loss, metric_fn: Loss):
        """Evaluates the performance of the model in the given metrics/losses.
        Predicts on an input and then uses output and compared to true values.
        ---------
        Arguments:
            - inputs (Array): A JAX compatible array that can be fed into the model for outputs.
            - labels (Array): A JAX compatible array that contains truth values.
            - loss_fn (Loss): A ``tensorwrap.nn.losses.Loss`` subclass that computes the loss of the predicted arrays.
            - metric_fn (Loss): A ``tensorwrap.nn.losses.Loss`` subclass that computes a human interpretable version of loss from the arrays.
        """
        pred = self.predict(inputs)
        metric = metric_fn(labels, pred)
        loss = loss_fn(labels, pred)
        self.__show_loading_animation(1, 1, loss, metric)

    def __show_loading_animation(self, total_batches, current_batch, loss, metric):
        """Helper function that shows the loading animation, when training the model.

        NOTE: Private method.
        """
        length = 30
        filled_length = int(length * current_batch // total_batches)
        bar = colored('─', "green") * filled_length + '─' * (length - filled_length)
        print(f'\r{current_batch}/{total_batches} [{bar}]    -    loss: {loss}    -    metric: {metric}', end='', flush=True)


# Sequential models that create Forward-Feed Networks:
class Sequential(Model):
    def __init__(self, layers: list = list(), name="Sequential") -> None:
        super().__init__(name=name)
        self.layers = layers
        for layer in self.layers:
            self.add_module_params(layer)


    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
        self.add_module_params(layer)

    def call(self, params: dict, x: Array) -> Array:
        for layer in self.layers:
            x = layer(params, x)
        return x


# Inspection Fixes:
Model.__module__ = "tensorwrap.nn.models"
Sequential.__module__ = "tensorwrap.nn.models"