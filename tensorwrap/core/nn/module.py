import tensorflow as tf
from tensorflow import (function,
                        Variable,
                        Tensor)
from collections import defaultdict
from typing import Optional

class Module(tf.Module):
  """Base class for all neural network modules.
  
  Your models should also subclass this class.
  
  Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign
  the submodules as regular attributes.
  
  Example:
  ```python
  from tensorwrap import nn
  from tensorwrap.nn import functional as F
  
  class Model(nn.Module):
    def __init__(self):
      self.conv1 = nn.Conv2d(1, 20, 5)
      self.conv2 = nn.Conv2d(20, 20, 5)
    
    def forward(self, x):
      x = F.relu(self.conv1(x))
      return F.relu(self.conv2(x))
  ```
  
  Arguments:
    dynamic (bool, optional): Specifies whether to compile ``forward`` method or not. Defaults to True.
    name (str, optional): Specifies the name of the module class.
  """
  _name_setter_dict = defaultdict(int)

  def __init__(self, name: Optional[str | None] = None):
    # Handling Distinct Names:
    name = name or self.__class__.__name__
    name_id = Module._name_setter_dict[name]
    
    if not isinstance(name, str):
      raise TypeError(f"Name attribute must be a string value, not {type(name)}")
    elif name_id:
      name = f"{name}_{name_id}"
    
    Module._name_setter_dict[name] += 1
    super().__init__(name)
    
    # Setting training attributes
    self.training = False
    

  # Training and Evalutation Methods:
  def train(self, mode: Optional[bool] = True):
    """Sets the module in training mode.
    
    This only affects certain modules, e.g. ``Dropout``, ``BatchNorm``, etc.
    
    Arguments:
      mode (bool, optional) - Specifies whether to set training mode (True) or evaluation mode (False).
                              Defaults to True.
    """
    if not isinstance(mode, bool):
      raise TypeError(f"Raised from {self.name}"
                      f"Argument mode must be type boolean, not {type(mode)}")
    self.training = mode

  def eval(self):
    """Sets the module in evaluation mode.
    
    This only affects certain modules, e.g. ``Dropout``, ``BatchNorm``, etc.
    
    Equivalent with self.train(False)."""
    
    self.train(False) # Calling train function for reduced maintainence

  def compile(self, compile: Optional[bool] = True, *args, **kwargs):
      """Compiles/uncompiles this Module's forward using ``tf.function``.
      
      If this Module's forward is compiled, then all extra arguments are passed as-is to
      ``tensorflow.function``. Check ``tensorflow.function``'s documentation to decide additional arguments.
      
      Arguments:
        compile (bool, optional): Specifies whether to compile or uncompile the forward function. Defaults to True."""
      if not isinstance(compile, bool):
        raise TypeError(f"Raised from {self.name}"
                        f"Argument mode must be type boolean, not {type(compile)}")
      try:
        self.forward = function(self.forward, *args, **kwargs)  if compile else self.forward.python_function
      except AttributeError:
          print("Skipping uncompilation, since model wasn't compiled to begin with.")

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def forward(self, x):
    pass
  
  def __repr__(self) -> str:
    repr_string = f"{self.name}(\n"
    for key, val in vars(self).items():
        if isinstance(val, Module):
            new_string = f"  ({key}): {val.__repr__()}\n"
            repr_string += new_string
    repr_string += ')'
    return repr_string


def Parameter(tensor: Tensor, requires_grad: Optional[bool] = True):
  """A kind of Tensor that is to be considered a module parameter.

  In PyTorch, Parameters are Tensor subclasses, that have a very special property when used with ``Module``s -
  when they’re assigned as Module attributes they are automatically added to the list of its
  parameters, and will appear e.g. in parameters() iterator. Assigning a Tensor doesn’t have such 
  effect. This is because one might want to cache some temporary state, like last hidden state of 
  the RNN, in the model. If there was no such class as Parameter, these temporaries would get 
  registered too. 
  
  However, TensorFlow treats ``Variable`` tensors as ``Parameter`` equivalent, thus making this
  only a placeholder function that wraps around a trainable tensor.

  Arguments:
    data (Tensor): parameter tensor
    requires_grad (bool, optional) - Specifies if the paramter requires gradient. Defaults to True.
  """
  return Variable(tensor, trainable=requires_grad)