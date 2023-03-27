""" This is the activation's module for TensorWrap"""

import tensorwrap as tf
from tensorwrap.module import Module

class Activation(Module):
    def __init__(self):
        super().__init__()
        pass
    
    @classmethod
    def get(self, name):
        self.dict = {
            'relu': ReLU,
        }
        try:
            return self.dict[name]
        except:
            raise ValueError(f"The activation function {name} doesn't exist.")

    def call(self):
        pass

    def __call__(self, inputs):
        return self.call(inputs)

class ReLU(Activation):
    def __init__(self, 
                 max_value=None,
                 negative_slope = 0,
                 threshold = 0):
        super().__init__()
        self.max_value = max_value
        self.slope = negative_slope
        self.threshold = threshold

        if self.max_value is not None and self.max_value < 0:
            raise ValueError("Max_value cannot be negative.")
    
    def call(self, inputs):

        return tf.minimum(tf.maximum(0, inputs - self.threshold), self.max_value)