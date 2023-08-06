""" This is the Keras API of TensorWrap, which aims to offer a similar
API as tf.keras from TensorFlow. It contains neural network modules that are
contained in the original Keras API and aims to simplify computing and prototyping."""

# Integrated Libraries:
import optax as optimizers

# Import Libraries:
# from . import optimizers
from . import activations, callbacks, initializers, layers, losses, models
from .models.base import Model, Sequential

# Path Shorteners:
