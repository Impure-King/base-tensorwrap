""" This is the activation's module for TensorWrap"""

import tensorwrap as tf
from tensorwrap.module import Module

class Activation(Module):
    def __init__(self):
        super().__init__()
        pass
    
    @classmethod
    def get(cls, name):
        cls.dict = {
            None : cls.no_activate,
            'relu': ReLU(),
        }
        try:
            return cls.dict[name.lower()]
        except:
            raise ValueError(f"The activation function {name} doesn't exist.")

    @staticmethod
    def no_activate(inputs):
        return inputs
    def call(self, inputs):
        raise NotImplementedError("Please implement the call function to define control flow.")

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
        part1 = tf.maximum(0, inputs - self.threshold)
        if self.max_value is not None:
            return tf.minimum(part1, self.max_value)
        else:
            return part1