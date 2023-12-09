from typing import Any
import tensorwrap as tf
from tensorwrap.module import Module

class Initializer(Module):
    def __init__(self, name: str = "Initializer", *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)


class GlorotUniform(Initializer):
    def __init__(self, name: str = "GlorotUniform", *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)

    def call(self, shape):
        return tf.randu(shape)


class GlorotNormal(Initializer):
    def __init__(self, name: str = "GlorotNormal", *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)

    def call(self, shape):
        return tf.randn(shape)


class Zeros(Initializer):
    def __init__(self, name: str = "Zeros", *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)

    def call(self, shape):
        return tf.zeros(shape)
    
# Defining some modules:
GlorotUniform.__module__ = "tensorwrap.nn.initializers"
GlorotNormal.__module__ = "tensorwrap.nn.initializers"
Zeros.__module__ = "tensorwrap.nn.initializers"