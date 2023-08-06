""" TensorWrap's initial API. This API contains various frontend functions that are
used to manipulate tensors and various data. However, most of the API currently borrows
from JAX's built-in operations. For neural networks, please use the nn API or import
the Torch API to use PyTorch variants."""

# JAX Built-ins:
from jax import disable_jit as disable_jit
from jax import grad as grad
from jax import value_and_grad as value_and_grad
from jax import vmap as vectorized_map
from jax.numpy import abs as abs
from jax.numpy import arange as range
from jax.numpy import argmax as argmax
from jax.numpy import argmin as argmin
from jax.numpy import array as tensor
from jax.numpy import expand_dims as expand_dims
from jax.numpy import eye as identity
from jax.numpy import float16 as float16
from jax.numpy import float32 as float32
from jax.numpy import float64 as float64
from jax.numpy import matmul as matmul
from jax.numpy import max as max
from jax.numpy import maximum as maximum
from jax.numpy import mean as mean
from jax.numpy import min as min
from jax.numpy import minimum as minimum
from jax.numpy import ones as ones
from jax.numpy import prod as prod
from jax.numpy import reshape as reshape
from jax.numpy import shape as shape
from jax.numpy import square as square
from jax.numpy import squeeze as squeeze
from jax.numpy import sum as sum
from jax.numpy import zeros as zeros

# Library Paths:
from tensorwrap import config, experimental, nn, test
from tensorwrap.experimental.serialize import load_model, save_model
from tensorwrap.experimental.wrappers import function

# Path Shortener:
from tensorwrap.module import Module
from tensorwrap.ops import last_dim, randn, randu
from tensorwrap.version import __version__
