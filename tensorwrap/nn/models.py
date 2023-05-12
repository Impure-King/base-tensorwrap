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
        self.trainable_variables = {}
        self.__verbosetracker = (
            self.__verbose0,
            self.__verbose1,
            self.__verbose2
        )

        for i in vars(self).values():
            if isinstance(i, tf.nn.layers.Layer):
                self.trainable_variables[i.name] = i.trainable_variables
            try:
                for elem in i:
                    if isinstance(elem, tf.nn.layers.Layer):
                        self.trainable_variables[elem.name] = elem.trainable_variables
            except TypeError:
                pass


    def compile(self,
                loss,
                optimizer,
                metrics = None):
        """Used to compile the nn model before training."""
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else loss

    def train_step(self,
                   x,
                   y=None):
        def loss_fun(param, x = x, y = y):
            y_pred = self.__call__(x)
            loss = self.loss_fn(y, y_pred)
            return loss
        loss = loss_fun(self.trainable_variables)
        grads = jax.grad(loss_fun)(self.trainable_variables) 
        self.trainable_variables = self.optimizer.apply_gradients(grads, self.trainable_variables)
        return loss

    def fit(self,
            x,
            y,
            batch_size = 32,
            epochs=1,
            verbose = 1,
            hist_return=False):
        print_func = self.__verbosetracker[verbose]
        hist = {}
        for epoch in range(1, epochs+1):
            loss = self.train_step(x, y)
            y_pred = self.__call__(x)
            metric = self.metrics(y, y_pred)
            print_func(epoch=epoch, epochs=epochs, metric=1, loss=loss)
            hist[epoch] = (loss, metric)
        if hist_return:
            return hist
    
    # Various reusable verbose functions:
    def __verbose0(self, *args, **kwargs):
        return 0

    def __verbose1(self, epoch, epochs, metric, loss):
        print(f"Epoch {epoch}|{epochs} \n"
                f"[=========================]    Loss: {loss:10.5f}     Metric: {metric:10.5f}")
    
    def __verbose2(self, epoch, epochs, metric, loss):
        print(f"Epoch {epoch}|{epochs} \t\t\t Loss: {loss:10.5f}\t\t\t     Metric: {metric:10.5f}")

    def evaluate(self,
                 x,
                 y_true):
        prediction = self.__call__(x)
        metric = self.metrics(y_true, prediction)
        loss = self.loss_fn(y_true, prediction)
        self.__verbose1(epoch=1, epochs=1, metric = metric, loss = loss)

    # Add a precision counter soon.
    def predict(self, x: Array, precision = None):
        try:
            array = self.__call__(x)
        except TypeError:
            x = jax.numpy.array(x, dtype = jax.numpy.float32)
            array = self.__call__(x)
        return array

    def call(self):
        pass

    def __call__(self, input, *args) -> Array:
        outputs = self.call(input)
        return outputs


class Sequential(Model):
    def __init__(self, layers=None) -> None:
        self.layers = [] if layers is None else layers
        super().__init__()


    def add(self, layer):
        self.layers.append(layer)

    
    def call(self, x) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x
