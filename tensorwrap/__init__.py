""" TensorWrap's initial API. This API contains various frontend functions that are
used to manipulate tensors and various data. However, most of the API currently borrows
from JAX's built-in operations. For neural networks, please use the Keras API or import
the Torch API to use PyTorch variants."""

# Error Silencer:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Library Paths:
from tensorwrap import keras

# Path Shorteners:
from tensorwrap.module import Module

# JAX Built-ins:
from jax.numpy import array as Variable
