# Stable Modules
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Iterable
from typing import final, Optional, Any, Callable

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


# All classes allowed for export.
__all__ = ["Module"]

class _MetaModule:
  def __new__(mcs, name, bases, attrs):
    # Check if the required method is present
    if 'call' not in attrs:
        raise TypeError(f"`call` method must be implemented to define control flow of the model.")
    return super().__new__(mcs, name, bases, attrs)

@register_pytree_node_class
class Module(dict):
  
  # A name tracking dictionary.
  _name_tracker:dict = defaultdict(int)
  
  def __init__(self, name="Module", dynamic=True):
    """Initiated a ``Module`` class object that can store trainable variables and operate on JAX transformations.
    Normally, it should be used as a subclass to register custom classes as pytrees and allow them to operate with
    JAX operations.
    
    Arguments:
      name (str): A string name that can be used to differentiate between multiple instances."""
    
    # Creating trainable attributes:
    self._no_grad:set = set()
    self._training_mode:bool = True
    self.trainable:bool = True
    self.built:bool = False
    self._init:bool = False

    # Managing Names:
    self.name:str = name + f":{Module._name_tracker[name]}"

  # Basic class customizations:
  def __init_subclass__(cls) -> None:
    """Used to register all subclasses as PyTrees, upon definition."""
    register_pytree_node_class(cls)

  def __getattr__(self, __key:str):
    """Manually customizing the ``__getattr__`` to allow for object oriented attribute query"""
    
    # Checking if instance has the key defined:
    if __key in self.keys():
      return self[__key]
    else:
      raise ValueError(f"{self.name} doesn't have attribute {__key!r}")

  def __setattr__(self, __key:str, __val:Any):
    """Manually customizing the ``__setattr__`` to allow for object oriented assignment."""
    self[__key]:Any = __val

  # Various Helper Functions for ``filter_and_map``:
  @staticmethod
  def is_module(module, key:str, value:Any):
    return isinstance(value, Module)
  
  @staticmethod
  def valid_children_filter(module, key:str, value:Any):
    return isinstance(value, (dict, list))
  
  @staticmethod
  def valid_parameter_filter(module, key:str, value:Any):
    return isinstance(value, (list, dict, jax.Array)) and not key.startswith('_')
  
  @staticmethod
  def valid_trainable_filter(module, key:str, value: Any):
    return Module.valid_parameter_filter(module, key, value) and not (key in module._no_grad)

  # Basic filtering algorithms:
  def basic_filter_and_map(self,
                           filter_fn,
                           map_fn = None):
    """An helper filter function to find children and aux_data."""
    if map_fn:
      map_fn = map_fn
    else:
      map_fn = lambda x: x
    true_dict = {}
    failed_dict = {}
    for attr_name in self.keys():
      if filter_fn(self, attr_name, self[attr_name]):
        true_dict[attr_name] = self[attr_name]
      else:
        failed_dict[attr_name] = self[attr_name]
    return map_fn(true_dict), failed_dict
  

  def filter_and_map(
      self,
      filter_fn: Callable[["tensorwrap.Module", str, Any], bool], # Basically a callable that takes ["tensorwrap.Module", str, Any] and returns a bool
      map_fn: Optional[Callable] = None,
      is_leaf_fn: Optional[Callable[["tensorwrap.Module", str, Any], bool]] = None
  ):
    """Recursively filter the contents of the module using ``filter_fn``,
    namely only select keys and values where ``filter_fn`` returns true.

    This is used to implement :meth:`parameters` and :meth:`trainable_parameters`
    but it can also be used to extract any subset of the module's parameters.

    Arguments:
        filter_fn (Callable): Given a value, the key in which it is found
            and the containing module, decide whether to keep the value or
            drop it.
        map_fn (Callable, optional): Optionally transform the value before
            returning it.
        is_leaf_fn (Callable, optional): Given a value, the key in which it
            is found and the containing module decide if it is a leaf.

    Returns:
        A dictionary containing the contents of the module recursively filtered
    """

    # Dealing with optional variables:
    map_fn = map_fn or (lambda x: x)
    is_leaf_fn = is_leaf_fn or (
      lambda m, k, v: not isinstance(v, (Module, dict, list))
    )

    def unwrap(vk, v):
      if is_leaf_fn(self, vk, v):
        return map_fn(v)
      
      if isinstance(v, Module):
        return v.filter_and_map(filter_fn, map_fn, is_leaf_fn)
      
      if isinstance(v, dict):
        nd = {}
        for k, v in v.items():
          tk = f"{vk}.{k}"
          nd[k] = unwrap(tk, v) if filter_fn(self, tk, v) else {}
        return nd
      
      if isinstance(v, list):
        nl = []
        for i, vi in enumerate(v):
          tk = f"{vk}.{i}"
          nl.append(unwrap(tk, vi) if filter_fn(self, tk, vi) else {})
        return nl
      
      raise RuntimeError("Unexpected leaf found while traversing the module.")
    
    return {k: unwrap(k, v) for k, v in self.items() if filter_fn(self, k, v)}
  
  # Variable Freezing and Manipulations:
  def freeze(self, __non_trainable_key:str):
    """Prevents the gradient computation of the specified variables.
    
    Arguments:
      __non_trainable_key (str): The string attribute name of the weight that is to be frozen."""
    
    if __non_trainable_key in self.trainable_variables:
      self._no_grad.add(__non_trainable_key)
    
    elif __non_trainable_key in self._no_grad:
      raise ValueError(f"{__non_trainable_key} is already frozen.")
    
    else:
      raise ValueError(f"{__non_trainable_key} isn't a valid variable.")

  def unfreeze(self, __trainable_key:str):
    """Allows the gradient computation of the specified variables.
    
    Arguments:
        __trainable_key (str): The string attribute name of the weight that is to be unfrozen."""
    
    if __trainable_key in self._no_grad:
      self._no_grad.remove(__trainable_key)

    elif __trainable_key in self.trainable_variables:
      raise ValueError(f"{__trainable_key} isn't currently frozen and it is currently trainable.")
    
    else:
      raise ValueError(f"{__trainable_key} isn't a valid variable.")
    
  # Extra Properties:
  @property
  def training(self):
    return self._training_mode
  
  # Variable Getters:
  @property
  def variables(self):
    return self.filter_and_map(self.valid_parameter_filter)
  
  @property
  def trainable_variables(self):
    return self.filter_and_map(self.valid_trainable_filter)
  
  @property
  def weights(self):
    return self.filter_and_map(self.valid_parameter_filter)

  @property
  def trainable_weights(self):
    return self.filter_and_map(self.valid_trainable_filter)


  # Initiation:
  def init(self, inputs, *args, **kwargs):
    self._init = True  
    with jax.disable_jit():
      self.__call__(inputs, *args, **kwargs)

  # Nice Repr for presentation:
  def __repr__(self) -> str:
    test = f'{self.name}'
    for attr_name, attr_val in self.variables.items():
      test += '\n'
      test += f"{attr_name}:{attr_val}"
    return test
  
  # Customizing the class initialization:
  def build(self, input=None):
    self.built = True
  
  def tree_values(self):
    return jax.tree_leaves(self)

  @jax.jit
  def __call__(self, inputs):
    if not self.built:
      self.build(inputs)

    return self.call(inputs)
  
  

  # Pytree Definition:
  def tree_flatten(self):
    true_dict, aux_data = self.basic_filter_and_map(self.valid_trainable_filter)
    aux_data["children_keys"] = tuple(true_dict.keys())
    children = tuple(true_dict.values())
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple):
    prev_init = cls.__init__
    
    def __init__(self):
      self._no_grad = set()
      self._training_mode = True
    
    cls.__init__ = __init__
    instance = cls()
    cls.__init__ = prev_init
    for attr_name, attr_val in zip(aux_data["children_keys"], children):
      instance[attr_name] = attr_val
    
    # aux_data.pop("children_keys")
    instance.update(aux_data)

    return instance



# @register_pytree_node_class
# class _Module(metaclass=ABCMeta):
#     """ Helper module class.
    
#     This helper class is a named container that acts to transform all subclassed containers into
#     pytrees by appropriately defining the tree_flatten and tree_unflatten. Additionally, it 
#     defines the trackable trainable_variables for all the subclasses."""

#     def __init__(self) -> None:
#         """Helps instantiate the class and assign a self.trainable_variables to subclass."""
#         pass

#     @classmethod
#     def __init_initialize__(cls):
#         """An extremely dangerous method which empties our the __init__ method and then create an instance. 
#         After, repurposing the __init__ again, and it returns an instance with an empty init function.
#         DO NOT USE for external uses."""
        
#         # function to replace __init__ temporarily:
#         def init_rep(self):
#             self.trainable_variables = {}
#         prev_init = cls.__init__  # Storing __init__ functionality
        
#         # Emptying and creating a new instance
#         cls.__init__ = init_rep
#         instance = cls()

#         # Reverting changes and returning instance
#         cls.__init__ = prev_init

#         return instance

#     def __init_subclass__(cls) -> None:
#         """Used to convert and register all the subclasses into Pytrees."""
#         register_pytree_node_class(cls)

#     def call(self, *args, **kwargs):
#         """Acts as an abstract method to force all implementation to occur in the `call` method."""
#         pass

#     # Various JAX tree registration methods. Overriding is not allowed.
#     def tree_flatten(self):
#         leaves = {}
#         # for key in self.trainable_variables:
#         #     leaves[key] = self.trainable_variables[key]
        
#         # Removing trainable_variables:
#         aux_data = vars(self).copy()
#         # aux_data.pop("trainable_variables")

#         return leaves, aux_data

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         instance = cls.__init_initialize__()
#         instance.params = children
#         vars(instance).update(aux_data)
#         return instance


# class Module(_Module):
#     """A base class for all JAX compatible neural network classes to subclass from.
#     Allows for all subclasses to become a Pytree and
#     assigns special functions to implicitly track trainable parameters from other subclassed objects.
#     ---------
#     Arguments:
#         - name (Optional, str): A string consisting for the internal name for state management. Defaults to ``"Module"``. 
#     """
    
#     _name_tracker = defaultdict(int) # Automatically initializes the indices to 0.

#     def __init__(self, 
#                  name: Optional[str] = "Module",
#                  trainable: Optional[bool] = True):
#         super().__init__()

#         # Implicit Name tracking upon model creation. Replacing Tracker with name tracker with default dict eventually.
#         self.name = f"{name}:{str(Module._name_tracker[name])}"
#         Module._name_tracker[name] += 1
      

#         # Defining parameter handling:
#         self.params = {self.name: {}}
#         self.child_blocks = []
        
#         # Trainable handling:
#         self.trainable = trainable
#         self.__nontrainable_weights = []

#         # Implicit Variables:
#         self._built = False
#         self._init = False

#         # Parameter Error Handling:
#         if not isinstance(name, str):
#             raise TypeError(f"Raised from {self.name}.\n"
#                             "``name`` parameter is not type 'str'.\n"
#                             f"Current argument: {name}")
    
#     def add_module_params(self, obj : object, strict : Optional[bool] = True):
#         """Queues the addition of a ``Module`` subclass's variables to the class's variables. 
#         The addition occurs at the model initialization.
#         ---------
#         Arguments:
#             - obj (Module): The ``Module`` subclass, whose variables are to be collected.
#             - strict (Optional, boolean): Determines whether to thrown an error, when incorrect type is provided. Defaults to ``True``.
#         """
#         if isinstance(obj, Module) and hasattr(obj, "name"):
#             self.child_blocks.append(obj)

#         elif strict:
#             if not isinstance(obj, Module):
#                 raise TypeError(f"""Raised from {self.name}.
#                                 Object {obj} is not a ``Module`` subclass.
#                                 Object type: {type(obj)}""")

#             elif not hasattr(obj, "name"):
#                 raise AttributeError(f"""Raised from {self.name}.
#                                     ``Module`` subclass {obj} does not have attribute name.""")
#         else:
#             pass

#     def _add_module_params(self):
#         """A hidden helper method that parses through all the registered ``Module`` subclasses and adds their parameter to the current instance."""
#         for block in self.child_blocks:
#             self.params[self.name][block.name] = block.params[block.name]        

#     def init(self, inputs: Any):
#         """The initializing method that gathers and defines the model parameters.
#         It requires all the hidden modules to be registered by the ``add_module_params`` method, for proper initialization.
#         ---------
#         Arguments:
#             - inputs (Any): The inputs that the model will evaluate to initialize the parameters.
#         """
#         self._init = True
#         self._add_module_params()
#         with jax.disable_jit():
#             self.__call__(self.params, inputs)
#         self._add_module_params()
#         return self.params

#     def build(self, *args):
#         """A build method that is called during initialization."""
#         self._built = True

    
#     def __call__(self, params: dict, inputs: Any, *args, **kwargs):
#         """The main call attribute of all the Modules.
#         ---------
#         Arguments:
#             - params (dict): A dictionary containing the ``Modules`` parameters.
#             - inputs (Any): The inputs that are required for the output predictions.
#         """
#         if not self._built:
#             self.build(inputs)
#             params = self.params
#         if not self._init:
#             self.init(inputs)
#             params = self.params
        
#         for name in self.__nontrainable_weights:
#             params[self.name][name] = jax.lax.stop_gradient(params[self.name][name])
            
#         if not self.trainable:
#             params[self.name] = jax.lax.stop_gradient(params[self.name])
#         out = self.call(params[self.name], inputs, *args, **kwargs)
#         return out

#     def format_table(self, d, depth=0):
#         dicts = defaultdict(int)
#         lens = []
#         if not d:
#             return ""
        
#         table = "─"*100 + '\n'
#         for key, value in d.items():
#             indent = "   " * depth
#             if isinstance(value, dict):
#                 dicts[depth] += 1
#                 start = f"|{indent}{dicts[depth]}.{key}\n"
#                 table += start + '─' * 100 + '\n'
#                 table += self.format_table(value, depth + 1)
#             else:
#                 table += f"|{indent}{key}: {value}\n"
        
#         return table

#     @final
#     def __repr__(self):
#         table = self.format_table(jax.tree_map(lambda x: x.shape, self.params))
#         return table
