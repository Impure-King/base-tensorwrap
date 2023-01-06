from jax import tree_util
import jax
from functools import partial

@tree_util.register_pytree_node_class
class Module:
    """ A Module is the base class for all neuralnet objects. Enter any keyword argument
    and it will unpack and it will create new attributes in self."""

    dynamics = False
    def __init__(self, **var):

        # Getting Variables:
        self.dict = var
        for keys in var:
            setattr(self, keys, var[keys])

    if dynamics:
        def tree_flatten(self):
            children = []
            for keys in self.dict:
                children.append(self.dict[keys])
            aux_data = {"dynamic": self.dynamic}
            return (children, aux_data)
    
    else:
        def tree_flatten(self):
            children = []
            aux_data = self.dict # All the static variables
            return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

@tree_util.register_pytree_node_class
class Model(Module):
    """Base neural network Model. It is a named character to train and evaluate all the layers.
"""

    @property
    def weights(self):
        self.weights = []
        for layer in self.layers:
            self.weights.append(layer.trainable_variables)
        return jax.numpy.array(self.weights)
    
    def trainable_weights(self):
        trainable_kernels = []
        trainable_bias = []
        for layer in self.layers:
            trainable_kernels.append(layer.trainable_variables[0])
            trainable_bias.append(layer.trainable_variables[1])
        self.trainable_kernels = jax.numpy.array(trainable_kernels)
        self.trainable_bias = jax.numpy.array(trainable_bias)
        return self.trainable_kernels, self.trainable_bias

    def compile(self, loss, optimizer, accuracy = None):
        self.loss_fn = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        for layer in self.layers:
            layer.build([1])
    
    @partial(jax.jit, static_argnums = 0)
    def train_step(self, y_true, y_pred):
        derivative_fn = jax.grad(self.loss_fn)
        self.optimizer.compute_grads(derivative_fn, y_true, y_pred)
        weights = self.optimizer.apply_gradients(Model.trainable_weights(self))
        for layer in self.layers:
            layer.update(weights)

    def fit(self, Model, x_train, y_train, epochs = 1):
        for epoch in range(epochs):
            y_pred = Model.__call__(self, x_train)
            accuracy = self.accuracy(y_train, y_pred)
            Model.train_step(self, y_train, y_pred)
            print(f"Epoch {epoch} : Loss = {accuracy}")
    

