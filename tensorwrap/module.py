from typing import TYPE_CHECKING, Any
class Module:
    """This is the base class for all types of functions and components.
    This is going to be a static type component, in order to allow jit.compile
    from jax and
    """
    if TYPE_CHECKING:
        def __init__(self, *args, **kwargs) -> None:
            # This makes sure that constructor arguments are accepted.
            pass
        
        def __call__(self, *args: Any, **kwds: Any) -> Any:
            # Makes the Module class callable.
            pass
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        
        cls._dataclass_transform()
        