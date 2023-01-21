import tensorwrap as tf
from tensorwrap import keras
import jax


layer = tf.keras.layers.Dense(units=1)

array = tf.Variable([[1, 2, 3]], dtype = tf.float32)
def function(x):
    if x:
        return 'LOL'
    elif not x:
        return 'Hi'
    else:
        x += 1
    return x


result = jax.tree_map(lambda x: x + 1, layer)
result2 = jax.tree_map(lambda x: x, result)
print(jax.tree_util.tree_leaves(result2))
print(jax.tree_util.tree_leaves(layer))
print(layer(array))



