# Stable Modules:
import jax
import copy
import tensorwrap as tf
from jax import numpy as jnp
from typing import (Any,
                    Tuple,
                    final)
from jaxtyping import Array
from functools import partial

# Custom built Modules:
import tensorwrap as tf
from tensorwrap.module import Module
from tensorwrap.nn.layers.base import Layer


__all__ = ["Model", "Sequential"]

class Model(Module):
    """ Main superclass for all models and loads any object as a PyTree with training and inference features."""

    _name_tracker = 0

    def __init__(self, dynamic = False, dtype = jnp.float32) -> None:
        super().__init__()
        self._compiled = False
        self._initialized = False
        self.dtype = dtype
        

    def __layer_initializer(self, _object) -> None:
        if isinstance(_object, (tf.nn.layers.Layer, tf.nn.activations.Activation)):
            if _object.name == 'layer':
                _object.name += f' {Model._name_tracker}'
                Model._name_tracker += 1
            self.trainable_variables[_object.name] = _object.trainable_variables

    def call(self, params: dict, inputs: Any, *args, **kwargs) -> Any:
        pass
    
    def __call__(self, params: dict, inputs: Array, *args, **kwargs) -> Array:
        if not self._initialized:
            raise NotImplementedError("The model parameters have not been initialized using ``model.init``")

        outputs = self.call(params, inputs, *args, **kwargs)
        return outputs
    
    def compile(self,
                loss,
                optimizer,
                metrics = None):
        """Used to compile the nn model before training."""
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else loss
        self._compiled = True

        
        def complete_grad(params, x, y):
            y_pred = self.__call__(params, x)
            losses = self.loss_fn(y, y_pred)
            return losses, y_pred

        self._value_and_grad_fn = jax.value_and_grad(complete_grad, has_aux=True)
    
    def init(self, x):
        for attr_name in dir(self):
            _object = getattr(self, attr_name)
            if isinstance(_object, list):
                for i in _object:    
                    self.__layer_initializer(i)
            else:
                self.__layer_initializer(_object)
        self._initialized = True
        self.__call__(self.trainable_variables, x)


    def train_step(self,
                   params,
                   x_train,
                   y_train) -> Tuple[dict, Tuple[int, int]]:
        """ Notes:
            Avoid using when using new loss functions or optimizers.
                - This assumes that the loss function arguments are (y_true, y_pred)."""
        if not self._compiled:
            raise NotImplementedError("The model has not been compiled using ``model.compile``.")
        (losses, y_pred), grads = self._value_and_grad_fn(params, x_train, y_train)
        params = self.optimizer.apply_gradients(params, grads)
        return params, (losses, y_pred)

    def fit(self,
            x_train,
            y_train,
            epochs = 1):
        
        step = jax.jit(self.train_step)

        for i in range(1, epochs + 1):
            self.trainable_variables, (loss, pred) = step(self.trainable_variables, x_train, y_train)
            metric = self.metrics(y_train, pred)
            print(f"Epoch: {i} \t\t Loss: {loss:.5f} \t\t Metrics: {metric:.5f}")
        
    def predict(self, x):
        return self.__call__(self.trainable_variables, x)



# Sequential models that create Forward-Feed Networks:
class Sequential(Model):
    def __init__(self, layers: list = []) -> None:
        self.layers = layers
        super().__init__()


    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def call(self, params: dict, x: Array) -> Array:
        for layer in self.layers:
            x = layer(params[layer.name], x)
        return x


# Inspection Fixes:
Model.__module__ = "tensorwrap.nn.models"
Sequential.__module__ = "tensorwrap.nn.models"