"""TensorWrap is a high level nueral net library that aims to provide prebuilt models,
layers, and losses on top of JAX. It aims to allow for faster prototyping, intuitive solutions,
and a coherent workflow while maintaining the benefits/compatibility with JAX.

With the expansion of the project, TensorWrap will also be able to develop a production system,
enabling JAX models to deploy outside of the python environment as well. Therefore, the current
version only supports prototyping and efficiency.
"""

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
from jax.numpy import concatenate as concat
from jax.numpy import dot as dot
from jax.numpy import expand_dims as expand_dims
from jax.numpy import eye as identity
from jax.numpy import float16 as float16
from jax.numpy import float32 as float32
from jax.numpy import float64 as float64
from jax.numpy import int4 as int4
from jax.numpy import int8 as int8
from jax.numpy import int16 as int16
from jax.numpy import int32 as int32
from jax.numpy import int64 as int64
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
from jax.numpy import stack as stack
from jax.numpy import sum as sum
from jax.numpy import zeros as zeros
from jax.numpy import asarray as convert_to_tensor

# Library Paths:

from tensorwrap import nn

# Fast Loading Modules:
from tensorwrap import config, experimental
from tensorwrap.experimental.serialize import load_model, save_model
from tensorwrap.experimental.wrappers import function

# Path Shortener:

from tensorwrap.ops import last_dim, randn, randu

# Fast Loading Modules:
from tensorwrap.module import Module
from tensorwrap.version import __version__
