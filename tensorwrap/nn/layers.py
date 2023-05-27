# Stable Modules:
import jax
import numpy as np
from jax import (jit,
                 numpy as jnp)
from jax.random import PRNGKey
from jaxtyping import Array
from random import randint
from typing import (Any,
                    Tuple,
                    final)

# Custom built Modules:
import tensorwrap as tf
from tensorwrap.module import Module

# Custom Trainable Layer


class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers, to ensure that they are converted in PyTrees."""

    name_tracker: int = 0

    def __init__(self, name: str = "layer", dynamic = False, trainable: bool = True, *args, **kwargs) -> None:
        super().__init__()
        self.built = False
        self.name = name + str(Layer.name_tracker)
        self.trainable = trainable
        self.dynamic = dynamic
        # Adding a defined out shape for all layers:
        # self.out_shape
        Layer.name_tracker += 1

    def add_weights(self, shape: Tuple[int, ...], key = PRNGKey(randint(1, 10)), initializer = 'glorot_normal', name = 'unnamed weight', trainable=True):
        """Useful method inherited from layers.Layer that adds weights that can be trained.
        ---------
        Arguments:
            - shape: Shape of the inputs and the units
            - initializer: The initial values of the weights
            - name: The name of the weight.
            - trainable (Optional) - Not required or implemented yet. """
        if initializer == 'zeros':
            weight = jnp.zeros(shape, dtype=jnp.float32)

        elif initializer == 'glorot_normal':
            weight = jax.random.normal(key, shape, dtype=tf.float32)

        elif initializer == 'glorot_uniform':
            weight = jax.random.uniform(key, shape, dtype=tf.float32)
        else:
            raise ValueError("Incorrect initializer is given.")

        # Adding to the trainable variables:
        if trainable:
            self.trainable_variables[name] = weight
        return weight


    def __repr__(self) -> str:
        return self.name

    @final
    def __call__(self, params: dict, inputs: Array):
        # This is to compile if not built.
        if not self.built:
            self.build(inputs)
            if not self.dynamic:
                self.__call = jax.jit(self.call)
        out = self.__call(params, inputs)
        return out
    
    
    def call(self, params: dict, inputs: Array):
        raise NotImplementedError("Call Method Missing:\nPlease define the control flow in the call method.")

    # Needed to make the layer built
    def build(self, inputs: Array = None):
        self.built = True


# Dense Layer:

class Dense(Layer):
    """ A fully connected layer that applies linear transformation to the inputs.

    Args:
        units (int): A positive integer representing the output shape.
        activation (Optional, str or Activation): Activation function to use. Defaults to None.
        use_bias (Optional, bool): A boolean signifying whether to include a bias term.
        kernel_initializer (Optional, str or Initializer)
    """

    name_tracker: int = 0

    def __init__(self,
                 units: int,
                 use_bias: bool = True,
                 kernel_initializer: Module = 'glorot_uniform',
                 bias_initializer: Module = 'zeros',
                 dynamic: bool = False,
                 *args,
                 **kwargs):
        super().__init__(dynamic=dynamic, *args, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name = 'dense ' + str(Dense.name_tracker)
        Dense.name_tracker += 1
        Layer.name_tracker -= 1

    def build(self, input_shape: int):
        input_shape = tf.last_dim(input_shape)
        self.kernel = self.add_weights(shape = (input_shape, self.units),
                                       initializer = self.kernel_initializer,
                                       name = "kernel")
        if self.use_bias:
            self.bias = self.add_weights(shape = (self.units,),
                                         initializer = self.bias_initializer,
                                         name="bias")
        else:
            self.bias = None
        super().build()
    
    def call(self, params: dict, inputs: Array) -> Array:
        x = inputs @ params['kernel'] + params['bias']
        return x


# Non-trainable Layers:

# class Lambda(Module):
#     """A non-trainable layer that applies a callable to the input tensor.

#     This layer is useful for applying custom functions or operations to the input tensor
#     without introducing any trainable variables. Additionally, it acts as a superclass for custom
#     nontrainable layers.

#     Args:
#         func (callable): The function or operation to apply to the input tensor. Defaults to None.

#     Example 1:
#         >>> def add_one(x):
#         ...     return x + 1
#         >>> layer = Lambda(add_one)
#         >>> layer(torch.tensor([1, 2, 3]))
#         tensor([2, 3, 4])

#     Example 2:
#         >>> import tensorwrap as tf
#         >>> class Flatten(tf.nn.Lambda):
#         ...     def __init__(self):
#         ...         super().__init__()
#         ...     
#         ...     def call(self, inputs):
#         ...         batch_size = tf.shape(inputs)[0]
#         ...         input_size = tf.shape(inputs)[1:]
#         ...         output_size = tf.prod(tf.Variable(input_shape))
#         ...         return tf.reshape(inputs, (batch_size, output_size))
#         >>> x = tf.range(1, 1e5)
#         >>> x = tf.range(1, int(1e5))
#         >>> x = tf.reshape(x, (1, 3, 11111, 3))
#         >>> Flatten()(x).shape
#         (1, 99999)

#     Inherits from:
#         Module

#     """
#     __name_tracker = 0
#     def __init__(self, func: Any = None, **kwargs):
#         super().__init__()
#         self.func = func
#         self.name = "lambda" + str(Lambda.__name_tracker)
#         self.trainable_variables = {}


    
#     def __call__(self, params, inputs):
#         return self.call(inputs)

#     def call(self, inputs):
#         """Applies the callable to the input tensor.

#         Args:
#             inputs: The input tensor.

#         Returns:
#             The output tensor after applying the callable to the input tensor.
#         """
#         return self.func(inputs) if not self.func == None else inputs

# # Flatten Layer:


# class Flatten(Lambda):
#     """
#     A layer that flattens the input tensor, collapsing all dimensions except for the batch dimension.

#     Args:
#         input_shape (Optional, Array): A tuple specifying the shape of the input tensor. If specified, the layer will use this shape to determine the output size. Otherwise, the layer will compute the output size by flattening the remaining dimensions after the batch dimension.

#     Example:
#         >>> # Create a Flatten layer with an input shape of (None, 28, 28, 3)
#         >>> flatten_layer = tf.nn.layers.Flatten()
#         ...
#         >>> # Apply the Flatten layer to a tensor of shape (None, 28, 28, 3)
#         >>> y = flatten_layer(x)
#         >>> print(y.shape)
#         (None, 2352)

#     Inherits from:
#         Module
#         Lambda
#     """
#     __name_tracker = 0
#     def __init__(self, input_shape=None, name = "flatten"):
#         super().__init__()
#         self.shape = input_shape
#         self.name = name + str(Flatten.__name_tracker)

#     def call(self, inputs):
#         """
#         Flattens the input tensor, collapsing all dimensions except for the batch dimension.

#         Args:
#             inputs: Input tensor.

#         Returns:
#             Flattened tensor with shape (batch_size, output_size).
#         """
#         batch_size = tf.shape(inputs)[0]
#         if self.shape is not None:
#             output_size = self.shape
#         else:
#             input_shape = tf.shape(inputs)[1:]
#             output_size = np.prod(np.array(input_shape))
#         return np.reshape(inputs, (batch_size, output_size))


# class Concat(Lambda):
#     """
#     A layer that concatenates all arrays given.

#     Example:
#         >>> # Create a concat layer:
#         >>> concat = tf.nn.layers.Concat()
#         ...
#         >>> # Creating two arrays to concatenate:
#         >>> x = tf.Variable([1, 2, 3])
#         >>> y = tf.Variable([4, 5, 6])
#         >>> # Apply the Flatten layer to all the tensors:
#         >>> z = concat(x, y)
#         >>> print(z)
#         [1, 2, 3
#          4, 5, 6]

#     Inherits from:
#         Module
#         Lambda
#     """

#     def call(self, inputs, axis=0):
#         """
#         Flattens the input tensor, collapsing all dimensions except for the batch dimension.

#         Args:
#             inputs: Input tensors in a list with the same dimensions.

#         Returns:
#             A concatenated tensor, with all the elements.
#         """
#         if not isinstance(inputs, list):
#             raise ValueError(
#                 f"A list of tensors wasn't inputted. Instead, the input type is {type(inputs)}.")

#         try:
#             return jax.numpy.concatenate(inputs, axis=axis)
#         except TypeError:
#             raise ValueError(
#                 "The input tensors don't have homogenous dimensions.")
