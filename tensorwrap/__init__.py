""" TensorWrap's initial API. This API contains various frontend functions that are
used to manipulate tensors and various data. However, most of the API currently borrows
from JAX's built-in operations. For neural networks, please use the Keras API or import
the Torch API to use PyTorch variants."""

# Error Silencer:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Library Paths:
from tensorwrap import nn
from tensorwrap import test
from tensorwrap import config
from tensorwrap import experimental

# Path Shortener:
from tensorwrap.module import Module
from tensorwrap.version import __version__
from tensorwrap.experimental.serialize import save_model, load_model
from tensorwrap.experimental.wrappers import function
from tensorwrap.ops import (last_dim,
                            randu,
                            randn)

# JAX Built-ins:
from jax import (disable_jit as disable_jit,
                 grad as grad,
                 value_and_grad as value_and_grad,
                 vmap as vectorized_map)

from jax.numpy import (abs as abs,
                       arange as range,
                       argmax as argmax,
                       argmin as argmin,
                       array as tensor,
                       eye as identity,
                       expand_dims as expand_dims,
                       float16 as float16,
                       float32 as float32,
                       float64 as float64,
                       matmul as matmul,
                       max as max,
                       maximum as maximum,
                       mean as mean,
                       min as min,
                       minimum as minimum,
                       ones as ones,
                       prod as prod,
                       reshape as reshape,
                       shape as shape,
                       square as square,
                       squeeze as squeeze,
                       sum as sum,
                       zeros as zeros)