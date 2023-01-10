"""Torch's Neural Network Wrapper"""

from . import functional
from . import optim
from . import activations
from . import initializers
from . import losses


# Path Shorteners:

from .models.sequential import Sequential
from .models.modules import Module, Model
from .functional import Linear