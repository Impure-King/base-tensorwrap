import jax.numpy as jnp
import jax

key = jax.random.PRNGKey(0)

@jax.jit
def add_weight(
    self,
    shape = (1,),
    dtype = jnp.float32,
    initializer = None,
    regularizer = None,
    trainable = None,
    constraint = None,
    **kwargs):

    """Adds a new trainable variable to a layer.

    Args:
          name: Variable name.
          shape: Variable shape. Defaults to scalar if unspecified.
          dtype: The type of the variable. Defaults to `self.dtype`.
          initializer: Initializer instance (callable).
          regularizer: Regularizer instance (callable).
          trainable: Boolean, whether the variable should be part of the layer's
            "trainable_variables" (e.g. variables, biases)
            or "non_trainable_variables" (e.g. BatchNorm mean and variance).
            Note that `trainable` cannot be `True` if `synchronization`
            is set to `ON_READ`.
          constraint: Constraint instance (callable).
          **kwargs: Additional keyword arguments. Accepted values are `getter`,
            `collections`, `experimental_autocast` and `caching_device`.
        Returns:
          The variable created.
        Raises:
          ValueError: When giving unsupported dtype and no initializer or when
            trainable has been set to True with synchronization set as
            `ON_READ`.
    """
    variable = jnp.array(jax.random.rand(key, shape = shape), dtype = jnp.float32)
    return variable

