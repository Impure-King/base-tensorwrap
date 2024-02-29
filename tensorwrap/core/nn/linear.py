from .module import Module, Parameter
from .initializers import glorot_uniform
import tensorflow as tf
from typing import Optional

class Linear(Module):
    """A Basic Linear Layer that computes a linear transformation.

    Arguments:
        in_features (int): Specifies the incoming shape of the inputs.
        out_features (int): Specifies the output shape.
        bias (bool, optional): Specifies whether to use bias (True) or not (False). Defaults to True.
    """
    def __init__(self, in_features:int, out_features:int, bias: Optional[bool] = True, **kwargs):
        super().__init__(**kwargs)
        k = 1/in_features
        self.weight = Parameter(glorot_uniform((out_features, in_features)))
        if bias:
            self.bias = Parameter(tf.zeros((out_features,)))
        
        self.use_bias = bias
        
        # Saving for repr
        self.in_features, self.out_features = in_features, out_features
    
    def forward(self, x):
        if str(x.dtype).find('float') == -1:
            raise TypeError(
                f"Raised from {self.name}.\n"
                f"Input tensor dtype ``tf.float32``. Current dtype {x.dtype}")
            
        if self.use_bias:
            return x @ tf.transpose(self.weight) + self.bias
        return x @ tf.transpose(self.weight)
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"