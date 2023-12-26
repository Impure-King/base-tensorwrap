# Stable Modules:
import jax
from jax import numpy as jnp
from jaxtyping import Array
from termcolor import colored
from typing import Any
import optax

# Custom built Modules:
import tensorwrap as tw
from tensorwrap.module import Module
from tensorwrap.nn.layers.base_layers import Layer
from tensorwrap.nn.losses.base import Loss
from tensorwrap.experimental import Dataloader


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

    def __init__(self, name: str = "Model", *args, **kwargs) -> None:
        super().__init__(name=name,
                         *args,
                         **kwargs) # Loads Module configurations
    
    
    def predict(self, inputs: jax.Array) -> jax.Array:
        """Returns the predictions, when given inputs for the model.
        
        Arguments:
            - inputs: Proprocessed JAX arrays that can be used to calculate an output."""
        self.__show_loading_animation(1, 1, None, None)
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
    
    def to(self, device_name: str):
        """Shifts the parameters and operations of the model to the suggested devices.
        Arguments:
            - device_name (str): A string that specifies device name."""
        jax.tree_map(lambda x: tw.config.device_put(x, device_name), self.weights)

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
    def __init__(self, layers: list = list(), name="Sequential", *args, **kwargs) -> None:
        super().__init__(name=name,
                         *args,
                         **kwargs)
        self.layers = layers


    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def call(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


# Inspection Fixes:
Model.__module__ = "tensorwrap.nn.models"
Sequential.__module__ = "tensorwrap.nn.models"