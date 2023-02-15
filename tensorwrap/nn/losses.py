from tensorwrap.module import Module
import tensorwrap as tf
from jax import jit


def mse(y_true, y_pred):
    return tf.mean(tf.square(y_pred - y_true))


def mae(y_true, y_pred):
    return tf.mean(tf.abs(y_pred - y_true))
