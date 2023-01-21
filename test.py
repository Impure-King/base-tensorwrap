import tensorwrap as tf
from tensorwrap import keras
print(tf.__version__)

print(tf.test.is_gpu_available())

layer = keras.layers.Dense(units=1)

inputs = tf.Variable([[1, 2, 3]], dtype=tf.float32)

print(layer(inputs))
