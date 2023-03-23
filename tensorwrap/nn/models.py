"""Sets of functions that allow you to define a custom model or a Sequential model."""
import tensorwrap as tf
from tensorwrap.module import Module
import jax
from jaxtyping import Array
import copy

class Model(Module):
    """ Main superclass for all models and loads any object as a PyTree with training and inference features."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.trainable_variables = []
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
        """Used to compile the nn model before training."""
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metrics = metrics if metrics != None else loss

        # Creating different objects for all layers:
        for i in vars(self):
            _object = vars(self)[i]
            if isinstance(_object, tf.nn.layers.Layer):
                self.layers.append(_object)

        for layer in self.layers:
            self.layers.remove(layer)
            if layer in self.layers:
                self.layers.append(copy.deepcopy(layer))
            else:
                self.layers.append(layer)
        
        # Doesn't offer any speed ups:
        for i in range(len(self.layers)-1):
            self.layers[i+1].build(self.layers[i].units)
            self.trainable_variables.append(self.layers[i+1].trainable_variables)


    def train_step(self,
                   x,
                   y=None,
                   layer=None):
        self._y_pred = self.__call__(x)
        grads = jax.grad(self.loss_fn)(tf.mean(y), tf.mean(self._y_pred))
        self.layers = self.optimizer.apply_gradients(grads, layer)

    # Various reusable verbose functions:
    def __verbose0(self, epoch, epochs, y_true):
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
            epochs=1,
            verbose = 1):
        if verbose==0:
            print_func=self.__verbose0
        elif verbose==1:
            print_func=self.__verbose1
        else:
            print_func=self.__verbose2
        
        for epoch in range(1, epochs+1):
            self.train_step(x, y, self.layers)
            print_func(epoch=epoch, epochs=epochs, y_true=y)
            # print(self.trainable_variables)
    
    def evaluate(self,
                 x,
                 y_true):
        prediction = self.__call__(x)
        metric = self.metrics(y_true, prediction)
        loss = self.loss_fn(y_true, prediction)
        self.__verbose1(epoch=1, epochs=1, loss=loss, metric=metric)

    # Add a precision counter soon.
    def predict(self, x: Array, precision = None):
        try:
            array = self.__call__(x)
        except TypeError:
            x = jax.numpy.array(x, dtype = jax.numpy.float32)
            array = self.__call__(x)
        return array


class Sequential(Model):
    def __init__(self, layers=None) -> None:
        super().__init__()
        self.layers = [] if layers is None else layers

    def add(self, layer):
        self.layers.append(layer)


    def call(self, x) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x
