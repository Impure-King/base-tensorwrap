"""Sets of functions that allow you to define a custom model or a Sequential model."""
import jax
import copy
import tensorwrap as tf

from jaxtyping import Array
from functools import partial
from tensorwrap.module import Module

class Model(Module):
    """ Main superclass for all models and loads any object as a PyTree with training and inference features."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self._name_tracker = 0
        self.trainable_layers = {}

    def compile(self,
                loss,
                optimizer,
                metrics = None):
        """Used to compile the nn model before training."""
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else loss

        for attr_name in dir(self):
            _object = getattr(self, attr_name)
            if isinstance(_object, tf.nn.layers.Layer):
                if _object.name == 'layer':
                    _object.name = 'layer ' + str(self._name_tracker)
                    self._name_tracker += 1
                if _object in self.trainable_layers.values():
                    _object = copy.deepcopy(_object)
                self.trainable_layers[_object.name] = _object

    def train_step(self,
                   x,
                   y=None,
                   layer=None):
        self._y_pred = self.__call__(x)
        grads = jax.grad(self.loss_fn)(tf.mean(y), tf.mean(self._y_pred))
        self.trainable_layers = self.optimizer.apply_gradients(grads, layer)

    # Various reusable verbose functions:
    def __verbose0(self, *args, **kwargs):
        return 0

    def __verbose1(self, epoch, epochs, y_true):
        metric = self.metrics(y_true, self._y_pred)
        loss = self.loss_fn(y_true, self._y_pred)
        print(f"Epoch {epoch}|{epochs} \n"
                f"[=========================]    Loss: {loss:10.5f}     Metric: {metric:10.5f}")
    
    def __verbose2(self, epoch, epochs, y_true):
        metric = self.metrics(y_true, self._y_pred)
        loss = self.loss_fn(y_true, self._y_pred)
        print(f"Epoch {epoch}|{epochs} \t\t\t Loss: {loss:10.5f}\t\t\t     Metric: {metric:10.5f}")

    def fit(self,
            x,
            y,
            batch_size = 32,
            epochs=1,
            verbose = 1):
        if verbose==0:
            print_func=self.__verbose0
        elif verbose==1:
            print_func=self.__verbose1
        else:
            print_func=self.__verbose2
        
        for epoch in range(1, epochs+1):
            self.train_step(x, y, self.trainable_layers)
            print_func(epoch=epoch, epochs=epochs, y_true=y)
    
    def evaluate(self,
                 x,
                 y_true):
        prediction = self.__call__(x)
        metric = self.metrics(y_true, prediction)
        loss = self.loss_fn(y_true, prediction)
        self.__verbose1(epoch=1, epochs=1, y_true=y_true)

    # Add a precision counter soon.
    def predict(self, x: Array, precision = None):
        try:
            array = self.__call__(x)
        except TypeError:
            x = jax.numpy.array(x, dtype = jax.numpy.float32)
            array = self.__call__(x)
        return array

    def call(self) -> Array:
        pass

    def __call__(self, *args) -> Array:
        inputs = args[0]
        outputs = self.call(inputs)
        return outputs


class Sequential(Model):
    def __init__(self, layers=None) -> None:
        super().__init__()
        self.layers = [] if layers is None else layers
        for _object in self.layers:
            if isinstance(_object, tf.nn.layers.Layer):
                if _object.name == 'layer':
                    _object.name = 'layer' + str(self._name_tracker)
                    self._name_tracker += 1
                if _object in self.trainable_layers.values():
                    _object = copy.deepcopy(_object)
                self.trainable_layers[_object.name] = _object


    def add(self, layer):
        self.layers.append(layer)

    def call(self, x) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x
