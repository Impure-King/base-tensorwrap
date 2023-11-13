# Stable Modules
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Iterable
from typing import final, Optional, Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


# All classes allowed for export.
__all__ = ["Module"]


@register_pytree_node_class
class _Module(metaclass=ABCMeta):
    """ Helper module class.
    
    This helper class is a named container that acts to transform all subclassed containers into
    pytrees by appropriately defining the tree_flatten and tree_unflatten. Additionally, it 
    defines the trackable trainable_variables for all the subclasses."""

    def __init__(self) -> None:
        """Helps instantiate the class and assign a self.trainable_variables to subclass."""
        pass

    @classmethod
    def __init_initialize__(cls):
        """An extremely dangerous method which empties our the __init__ method and then create an instance. 
        After, repurposing the __init__ again, and it returns an instance with an empty init function.
        DO NOT USE for external uses."""
        
        # function to replace __init__ temporarily:
        def init_rep(self):
            self.trainable_variables = {}
        prev_init = cls.__init__  # Storing __init__ functionality
        
        # Emptying and creating a new instance
        cls.__init__ = init_rep
        instance = cls()

        # Reverting changes and returning instance
        cls.__init__ = prev_init

        return instance

    def __init_subclass__(cls) -> None:
        """Used to convert and register all the subclasses into Pytrees."""
        register_pytree_node_class(cls)

    def call(self, *args, **kwargs):
        """Acts as an abstract method to force all implementation to occur in the `call` method."""
        pass

    # Various JAX tree registration methods. Overriding is not allowed.
    def tree_flatten(self):
        leaves = {}
        # for key in self.trainable_variables:
        #     leaves[key] = self.trainable_variables[key]
        
        # Removing trainable_variables:
        aux_data = vars(self).copy()
        # aux_data.pop("trainable_variables")

        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        instance = cls.__init_initialize__()
        instance.params = children
        vars(instance).update(aux_data)
        return instance


class Module(_Module):
    """A base class for all JAX compatible neural network classes to subclass from.
    Allows for all subclasses to become a Pytree and
    assigns special functions to implicitly track trainable parameters from other subclassed objects.
    ---------
    Arguments:
        - name (Optional, str): A string consisting for the internal name for state management. Defaults to ``"Module"``. 
    """
    
    _name_tracker = defaultdict(int) # Automatically initializes the indices to 0.

    def __init__(self, name: Optional[str] = "Module"):
        super().__init__()

        # Implicit Name tracking upon model creation. Replacing Tracker with name tracker with default dict eventually.
        self.name = f"{name}:{str(Module._name_tracker[name])}"
        Module._name_tracker[name] += 1
      

        # Defining parameter handling:
        self.params = {self.name: {}}
        self.child_blocks = []

        # Implicit Variables:
        self._built = False
        self._init = False

        # Parameter Error Handling:
        if not isinstance(name, str):
            raise TypeError(f"Raised from {self.name}.\n"
                            "``name`` parameter is not type 'str'.\n"
                            f"Current argument: {name}")
    
    def add_module_params(self, obj : object, strict : Optional[bool] = True):
        """Queues the addition of a ``Module`` subclass's variables to the class's variables. 
        The addition occurs at the model initialization.
        ---------
        Arguments:
            - obj (Module): The ``Module`` subclass, whose variables are to be collected.
            - strict (Optional, boolean): Determines whether to thrown an error, when incorrect type is provided. Defaults to ``True``.
        """
        if isinstance(obj, Module) and hasattr(obj, "name"):
            self.child_blocks.append(obj)

        elif strict:
            if not isinstance(obj, Module):
                raise TypeError(f"""Raised from {self.name}.
                                Object {obj} is not a ``Module`` subclass.
                                Object type: {type(obj)}""")

            elif not hasattr(obj, "name"):
                raise AttributeError(f"""Raised from {self.name}.
                                    ``Module`` subclass {obj} does not have attribute name.""")
        else:
            pass

    def _add_module_params(self):
        """A hidden helper method that parses through all the registered ``Module`` subclasses and adds their parameter to the current instance."""
        for block in self.child_blocks:
            self.params[self.name][block.name] = block.params[block.name]        

    def init(self, inputs: Any):
        """The initializing method that gathers and defines the model parameters.
        It requires all the hidden modules to be registered by the ``add_module_params`` method, for proper initialization.
        ---------
        Arguments:
            - inputs (Any): The inputs that the model will evaluate to initialize the parameters.
        """
        self._init = True
        self._add_module_params()
        with jax.disable_jit():
            self.__call__(self.params, inputs)
        self._add_module_params()
        return self.params

    def build(self, *args):
        """A build method that is called during initialization."""
        self._built = True


    def __call__(self, params: dict, inputs: Any, *args, **kwargs):
        """The main call attribute of all the Modules.
        ---------
        Arguments:
            - params (dict): A dictionary containing the ``Modules`` parameters.
            - inputs (Any): The inputs that are required for the output predictions.
        """
        if not self._built:
            self.build(inputs)
            params = self.params
        if not self._init:
            self.init(inputs)
            params = self.params
        out = self.call(params[self.name], inputs, *args, **kwargs)
        return out

    def format_table(self, d, depth=0):
        dicts = defaultdict(int)
        lens = []
        if not d:
            return ""
        
        table = "─"*100 + '\n'
        for key, value in d.items():
            indent = "   " * depth
            if isinstance(value, dict):
                dicts[depth] += 1
                start = f"|{indent}{dicts[depth]}.{key}\n"
                table += start + '─' * 100 + '\n'
                table += self.format_table(value, depth + 1)
            else:
                table += f"|{indent}{key}: {value}\n"
        
        return table

    @final
    def __repr__(self):
        table = self.format_table(jax.tree_map(lambda x: x.shape, self.params))
        return table


# class Module1(_Module):
#     """A base class for all JAX compatible neural network classes to subclass from.
#     Allows for all subclasses to become a Pytree and
#     assigns special functions to implicitly track trainable parameters from other subclassed objects.

#     Parameters:
#         - name (str): A string consisting for the internal name for state management. Defaults to "Module". """
    
#     _name_tracker = defaultdict(int) # Automatically initializes the indices to 0.

#     def __init__(self, name: str = "Module"):
#         super().__init__()

#         # Implicit Name tracking upon model creation. Replacing Tracker with name tracker with default dict eventually.
#         self.name = f"{name}:{str(Module._name_tracker[name])}"
#         Module._name_tracker[name] += 1
      

#         # Defining parameter handling:
#         self.params = {self.name: {}}

#         # Implicit Variables:
#         self._built = False
#         self._init = False

#         # Parameter Error Handling:
#         if not isinstance(name, str):
#             raise TypeError("``name`` parameter is not type 'str'.\n"
#                             f"Current argument: {name}")
    
#     def _recursive_iteration(self, obj):
#         for value in obj:
#             if not isinstance(value, Module) and not isinstance(value, (str, jnp.ndarray)):
#                 return self._recursive_iteration(value)
#             elif isinstance(value, dict):
#                 return self._recursive_iteration(value.values())
#             elif isinstance(value, Module):
#                 value._init_params()
#                 self.params[self.name][value.name] = value.params[value.name]
    
#     def _init_params(self):
#         """A helper function that recursively registers all child nodes and parameters,
#         by parsing the attributes of the class."""
#         # Start parsing the items of the class:
#         for attribute_value in vars(self).values():
#             if isinstance(attribute_value, Module) and hasattr(attribute_value, "params"):
#                 self.params[self.name][attribute_value.name] = attribute_value.params[attribute_value.name]
#                 if not attribute_value._init:
#                     attribute_value._init_params()
#                 self.params[self.name][attribute_value.name] = attribute_value.params[attribute_value.name]
            
#             elif isinstance(attribute_value, dict):
#                 self._recursive_iteration(attribute_value.values())

#             elif isinstance(attribute_value, Iterable) and not isinstance(attribute_value, (jnp.ndarray, str)):
#                 self._recursive_iteration(attribute_value)
        

#     def init(self, array):
#         self._init = True
#         self._init_params()
#         with jax.disable_jit():
#             self.__call__(self.params, array)
#         self._init_params()
#         return self.params
    
#     def build(self, *args):
#         """A build method that is called during initialization."""
#         self._built = True


#     def __call__(self, params, inputs, *args, **kwargs):
#         if not self._built:
#             self.build(inputs)
#             self._init_params()
#             params = self.params
#         if not self._init:
#             self.init(inputs)
#             params = self.params
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




# class Module1(_Module):
#     """A base class for all JAX compatible neural network classes to subclass from.
#     Allows for all subclasses to become a Pytree and
#     assigns special functions to implicitly track trainable parameters from other subclassed objects.

#     Parameters:
#         - name (str): A string consisting for the internal name for state management. Defaults to "Module". """
    
#     _name_tracker = defaultdict(int) # Automatically initializes the indices to 0.

#     def __init__(self, name: str = "Module"):
#         super().__init__()

#         # Implicit Name tracking upon model creation. Replacing Tracker with name tracker with default dict eventually.
#         self.name = f"{name}:{str(Module._name_tracker[name])}"
#         Module._name_tracker[name] += 1

#         # Keeping track of children implicitly in a later function, in order to extract params.
#         self._children = OrderedDict({self.name: {}})
#         self._list = []

#         # Defining parameter handling:
#         self.params = {self.name: {}}

#         # Implicit Variables:
#         self._init = False
#         self._built = False

#         # Parameter Error Handling:
#         if not isinstance(name, str):
#             raise TypeError("``name`` parameter is not type 'str'.\n"
#                             f"Current argument: {name}")

#     def register_children(self, child: object, strict_checking: Optional[bool] = False):
#         """A method that registers any child node.
#         Child nodes are nodes that inherit from Module subclass and
#         possess certain attributes like trainable variables + name.

#         Parameters:
#             - child (object): An object that subclasses the ``Module`` class and has ``name``.
#             - strict_checking (Optional, bool): A boolean to indicate whether to throw an error, when incorrect obj is passed.
#             Defaults to "False". """

#         # Checking for correct dtype:
#         if not isinstance(strict_checking, bool):
#             raise TypeError("``strict_checking`` parameter is not a boolean.\n"
#                             f"Current dtype: {type(strict_checking)}")

#         # Checking for subclassing:
#         if isinstance(child, Module):
#             # Adding to child nodes for track:
#             self._children[self.name][child.name] = child._children[child.name]
#             self._list.append(child)

#         elif strict_checking:
#             raise ValueError("``child`` parameter is not a ``Module`` subclass.\n"
#                              f"Current subclass {type(child)}")
#         else:
#             pass  # Ignoring the errors.

#     def register_parameter(self):
#         for child in self._list:
#             try:
#                 print(f"Registering {child.name} - {child}")
#                 child.register_parameter()  # Force register children parameters
#                 # Actual registration step:
#                 self.params[self.name][child.name] = child.params[child.name]
#                 print(f"Succeeded")
#             except:
#                 print("Failed.")

#     def _recursive_checker(self, obj):
#         """A helper function that recursively checks any iterators for Module's subclasses"""
#         if isinstance(obj, Iterable) and not isinstance(obj, (str, jnp.ndarray, dict)):
#             for item in obj:
#                 self._recursive_checker(item)
#         else:
#             self.register_children(obj, strict_checking=False)

#     def _init_params(self):
#         """A helper function that recursively registers all child nodes and parameters,
#         by parsing the attributes of the class."""
#         # Start parsing the items of the class:
#         for attribute_name, attribute_value in vars(self).items():
#             self._recursive_checker(attribute_value)
#         self.register_parameter()

#     def init(self, array):
#         self._init = True
#         self._init_params()
#         with jax.disable_jit():
#             self.__call__(self.params, array)
#         self._init_params()
#         return self.params
#     def build(self, *args):
#         """A build method that is called during initialization."""
#         self._built = True

#     # def _register(self, obj):
#     #     """Helper function for initiating parameters."""
#     #     if isinstance(obj, Iterable) and not isinstance(obj, (jnp.ndarray, str)):
#     #         for objs in obj:
#     #             self._register(objs)
#     #     else:
#     #         self.register_children(obj, strict=False)

#     # def init_params(self, inputs, *args):
#     #     self._init = True
#     #     with jax.disable_jit():
#     #         self.__call__(self.params, inputs, *args)
#     #     return self.params

#     # def register_children(self, child, strict=False):
#     #     """Appends a child element for recursive searching of parameters."""
#     #     if hasattr(child, 'name') and isinstance(child, Module):
#     #         self._children[child.name] = child
#     #
#     #     elif not isinstance(child, Module):
#     #         if strict:
#     #             raise ValueError(f"{child} is not a subclass of ``tensorwrap.nn.Module``."
#     #                              "All registered children must be ``tensorwrap.nn.Module`` subclasses.")
#     #         else:
#     #             pass
#     #     else:
#     #         raise AttributeError(f"{child} has no attribute ``name``."
#     #                              "Ensure ``name`` hasn't been overwritten.")

#     # def flatten_params(self):
#     #     """A method that uses children to register """
#     #     for child_name, child in self._children.items():
#     #         child.flatten_params()
#     #         self._params[self.name][child_name] = child._params[child_name]

#     def __call__(self, params, inputs, *args, **kwargs):
#         if not self._built:
#             self.build(inputs)
#         if not self._init:
#             self.init(inputs)
#             params = self.params
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
#         self.format_table(jax.tree_map(lambda x: x.shape, self.params))
#         return ""

#     # def get_trainable_params(self):
#     #     """Recursively flattens the trainable layers in all the ``tensorwrap.nn.Module`` modules."""
#     #     # Registers the children for variable flattening.
#     #     for attribute_name, attribute_value in vars(self).items():
#     #         self._register(attribute_value)
#     #     self.flatten_params()
#     #     return {self.name: self._params}

#     # @property
#     # def params(self):
#     #     for attribute_name, attribute_value in vars(self).items():
#     #         self._register(attribute_value)
#     #     self.flatten_params()
#     #     return {self.name: self._params}
#     #
#     # @params.setter
#     # def params(self, value):
#     #     self._params = value[self.name]
#     #
#     # @params.deleter
#     # def params(self):
#     #     del self._params
