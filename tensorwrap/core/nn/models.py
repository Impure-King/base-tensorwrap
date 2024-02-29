from .module import Module

class Sequential(Module):
    def __init__(self, *args:Module, **kwargs):
        super().__init__(**kwargs)
        self.layers:list = list(args)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def append(self, module):
        self.layers.append(module)
    
    def __repr__(self):
        repr_string = f"{self.name}(\n"
        for key, val in enumerate(self.layers):
            if isinstance(val, Module):
                new_string = f"  ({key}): {val.__repr__()}\n"
                repr_string += new_string
        repr_string += ')'
        return repr_string
    
    def __iter__(self):
        return iter(self.layers)