"""Sets of functions that allow you to define a custom model or a Sequential model."""
import tensorwrap as tf
from tensorwrap.keras.layers import Layer
import jax
from jaxtyping import Array
from termcolor import colored as c

class Model(Layer):
    """ Main superclass for all models and loads any object as a PyTree with training and inference features."""

    def __init__(self, dynamic=False, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.dynamic = dynamic

    def call(self) -> None:
        pass

    def __call__(self, *args):
        if not self.dynamic:
            function = jax.jit(self.call)
        else:
            function = self.call

        inputs = function(args[0])
        return inputs

    def compile(self,
                loss,
                optimizer,
                metrics):
        """Used to compile a keras model before training."""
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def creator(self):
        self.array1 = tf.Variable(self.trainable_variables[0])
        self.array2 = tf.Variable(self.trainable_variables[1])

    def train_step(self,
                   x,
                   y=None):
        y_pred = self.__call__(x)
        metric = self.metrics(y, y_pred)
        grads_fn = jax.grad(self.loss_fn)
        grads = grads_fn(y, y_pred)
        self.trainable_variables = self.optimizer.apply_gradients(grads, self.array1, self.array2)
        return metric

    def fit(self,
            x=None,
            y=None,
            epochs=1):
        for epoch in range(1, epochs+1):
            metric = self.train_step(x, y)
            print(f"Epoch {epoch} complete - - - - - -  Metrics: {metric}")




class Sequential(Model):
    def __init__(self, layers=None) -> None:
        super().__init__()
        self.layers = [] if layers is None else layers
        self.trainable_variables = []

    def add(self, layer):
        self.layers.append(layer)


    def call(self, x) -> Array:
        self.kernelses = []
        self.biases = []
        for layer in self.layers:
            if len(layer.trainable_variables) != 0:
                self.kernelses.append(layer.trainable_variables[0])
                self.biases.append(layer.trainable_variables[1])
            x = layer(x)
        self.trainable_variables = [self.kernelses, self.biases]
        self.creator()
        return x
