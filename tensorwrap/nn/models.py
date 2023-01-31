"""Sets of functions that allow you to define a custom model or a Sequential model."""
import tensorwrap as tf
from tensorwrap.module import Module
import jax
from jaxtyping import Array


class Model(Module):
    """ Main superclass for all models and loads any object as a PyTree with training and inference features."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.layers = []

    def call(self) -> Array:
        pass

    def __call__(self, *args) -> Array:
        inputs = args[0]
        outputs = self.call(inputs)
        return outputs

    def compile(self,
                loss,
                optimizer,
                metrics = None):
        """Used to compile a keras model before training."""
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics = metrics
        for i in range(len(self.layers) - 1):
            output_shape = self.layers[i].units
            self.layers[i+1].build([1, output_shape])


    def train_step(self,
                   x,
                   y=None,
                   layer=None):
        y_pred = self.__call__(x)
        metric = self.metrics(y, y_pred)
        grads = jax.grad(self.loss_fn)(tf.mean(y), tf.mean(y_pred))
        self.layers = self.optimizer.apply_gradients(grads, layer)
        return metric

    def predict(self,
                x):
        """ The method for predicting values. It is more concise than directly passing
        the inputs through the hidden __call__ function.
        Args:
         - x: The input to predict on."""
        y_pred = self.__call__(x)
        return y_pred
    
    def evaluate(self,
                x: Array,
                y: Array,
                loss = None,
                metrics = None):
        """ The method for evaluating the loss on test set. Losses and Metrics must be passed,
        if the model wasn't compiled through the fit method.
        Args:
         - x: The inputs to predict on.
         - y: The ground-truth values.
         - loss (optional): The loss function to evaluate on.
         - metrics (optional): The metrics to evaluate on.
        """
        if loss != None:
            self.loss = loss
        else:
            self.loss = self.loss_fn
        if metrics != None:
            self.metric_fn = metrics
        else:
            self.metric_fn = self.metrics
        y_pred = self.__call__(x)
        loss = self.loss(y, y_pred)
        metric = self.metric_fn(y, y_pred)
        print(f"1/1 [==============================] - loss: {loss} - metric {metric}")

    def fit(self,
            x=None,
            y=None,
            epochs=1):
        self.layers[0].build(x.shape)
        for epoch in range(1, epochs+1):
            metric = self.train_step(x, y, self.layers)
            print(f"Epoch {epoch}/{epochs}")
            print(f"1/1 [==============================] - loss: {metric}")

class Models(Module):
    """ Main superclass for all models and loads any object as a PyTree with training and inference features."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.layers = []

    def call(self) -> Array:
        pass

    def __call__(self, *args) -> Array:
        inputs = args[0]
        outputs = self.call(inputs)
        return outputs

    def compile(self,
                loss,
                optimizer,
                metrics = None):
        """Used to compile a keras model before training."""
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics = metrics
        for i in range(len(self.layers) - 1):
            output_shape = self.layers[i].units
            self.layers[i+1].build([1, output_shape])


    def train_step(self,
                   x,
                   y=None,
                   layer=None):
        y_pred = self.__call__(x)
        metric = self.metrics(y, y_pred)
        grads = jax.grad(self.loss_fn)(tf.mean(y), tf.mean(y_pred))
        self.layers = self.optimizer.apply_gradients(grads, layer)
        return metric

    def predict(self,
                x):
        """ The method for predicting values. It is more concise than directly passing
        the inputs through the hidden __call__ function.
        Args:
         - x: The input to predict on."""
        y_pred = self.__call__(x)
        return y_pred
    
    def evaluate(self,
                x: Array,
                y: Array,
                loss = None,
                metrics = None):
        """ The method for evaluating the loss on test set. Losses and Metrics must be passed,
        if the model wasn't compiled through the fit method.
        Args:
         - x: The inputs to predict on.
         - y: The ground-truth values.
         - loss (optional): The loss function to evaluate on.
         - metrics (optional): The metrics to evaluate on.
        """
        if loss != None:
            self.loss = loss
        else:
            self.loss = self.loss_fn
        if metrics != None:
            self.metric_fn = metrics
        else:
            self.metric_fn = self.metrics
        y_pred = self.__call__(x)
        loss = self.loss(y, y_pred)
        metric = self.metric_fn(y, y_pred)
        print(f"1/1 [==============================] - loss: {loss} - metric {metric}")

    def fit(self,
            x=None,
            y=None,
            epochs=1):
        self.layers[0].build(x.shape)
        for epoch in range(1, epochs+1):
            metric = self.train_step(x, y, self.layers)
            print(f"Epoch {epoch}/{epochs}")
            print(f"1/1 [==============================] - loss: {metric}")


class Sequential(Model):
    def __init__(self, layers=None) -> None:
        super().__init__()
        self.layers = [] if layers is None else layers
        self.trainable_variables = []

    def add(self, layer):
        self.layers.append(layer)


    def call(self, x) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x
