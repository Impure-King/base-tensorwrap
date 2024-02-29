import tensorflow as tf
import numpy as np

def __compute_fans(shape):
    if len(shape) < 2:
        raise ValueError("Number of dimensions can't be less than 2 for initializer fan computation.")
    fan_in = float(np.prod(shape[:-1]))
    fan_out = float(np.prod(shape[1:]))
    return fan_in, fan_out

# @tf.function
def glorot_uniform(shape:tuple, gain:float=1.0):
    """Returns an Tensor object that follows the Xavier uniform distribution."""
    fan_in, fan_out = __compute_fans(shape)
    a = gain * (6/(fan_in + fan_out))**0.5
    return tf.random.uniform(shape, minval=-1*a, maxval=a)

# @tf.function
def glorot_normal(shape:tuple, gain:float=1.0):
    """Returns an Tensor object that follows the Xavier normal distribution."""
    fan_in, fan_out = __compute_fans(shape)
    std = gain * (2/(fan_in + fan_out))**0.5
    return tf.random.normal(shape=shape, mean=0, stddev=std)
