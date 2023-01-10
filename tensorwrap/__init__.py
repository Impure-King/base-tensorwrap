'''TensorWrap API'''

# Silencing the warnings:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importing necessary requirements:
from . import core
from . import keras
import jax

# Extra path shorteners:

from functools import partial as alter
from tensorwrap.self import add_weight
from tensorwrap.core.tensors import expand_dims
from tensorwrap.core.tensors import range

# Jax method conversion:
from jax import numpy
from jax.numpy import float32
from jax.numpy import array as Variable
from jax.numpy import absolute
from jax import jit as function
from jax import disable_jit
from jax import random, grad

# Don't remove:
from .version import __version__