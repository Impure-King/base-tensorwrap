""" This is the activation's module for TensorWrap"""

import tensorwrap as tf
from tensorwrap.module import Module

class Activation(Module):
    __layer_tracker = 0
    def __init__(self, name = "activation"):
        super().__init__()
        self.name = name + str(Activation.__layer_tracker)
        self.id = int(self.name[-1])
        Activation.__layer_tracker += 1
    
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
    
    def init(self, input_shape):
        self._output_shape = input_shape

    def compute_output_shape(self):
        return self._output_shape

    @staticmethod
    def no_activate(inputs):
        return inputs
    def call(self, inputs):
        raise NotImplementedError("Please implement the call function to define control flow.")

    def __call__(self, params, inputs, *args, **kwargs):
        return self.call(inputs)

class ReLU(Activation):
    __name_tracker = 0
    def __init__(self, 
                 max_value=None,
                 negative_slope = 0,
                 threshold = 0,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)
        self.max_value = max_value
        self.slope = negative_slope
        self.threshold = threshold
        self.name = "relu " + str(ReLU.__name_tracker)
        ReLU.__name_tracker += 1
        if self.max_value is not None and self.max_value < 0:
            raise ValueError("Max_value cannot be negative.")
    
    def call(self, inputs):
        part1 = tf.maximum(0, inputs - self.threshold)
        if self.max_value is not None:
            return tf.minimum(part1, self.max_value)
        else:
            return part1