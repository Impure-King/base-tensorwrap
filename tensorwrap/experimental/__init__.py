"""This is a module of native TensorWrap API and different experimental features."""

from tensorwrap.experimental import wrappers
from tensorwrap.experimental import serialize
from tensorwrap.experimental.arrayDataLoader import Dataloader
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ['KERAS_BACKEND'] = 'jax'
from keras_core import datasets