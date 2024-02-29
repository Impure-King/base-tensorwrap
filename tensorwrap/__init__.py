"""TensorWrap is a high level nueral net library that aims to provide prebuilt models,
layers, and losses on top of JAX. It aims to allow for faster prototyping, intuitive solutions,
and a coherent workflow while maintaining the benefits/compatibility with JAX.

With the expansion of the project, TensorWrap will also be able to develop a production system,
enabling JAX models to deploy outside of the python environment as well. Therefore, the current
version only supports prototyping and efficiency.
"""

# Silencing tensorflow warnings:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import *
from tensorflow import constant as tensor
from tensorflow.experimental import numpy

numpy.experimental_enable_numpy_behavior(prefer_float32=True)

# Removing excess APIs:
exclude_api_names = [
    'nn',
    'config'
]

for api in exclude_api_names:
    globals().pop(api)

# Library Paths:
from tensorwrap import nn
from tensorwrap.core import config
from tensorwrap.core.ops import *


# Fast Loading Modules:
from tensorwrap.version import __version__
