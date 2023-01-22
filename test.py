import tensorwrap as tf
from tensorwrap import keras


layer = keras.layers.Dense(units=1000)
layer1 = keras.layers.Dense(units=1)
array = tf.Variable([[1, 2, 3]], dtype=tf.float32)

model = keras.Sequential()
model.add(layer)
model.add(layer)
model.add(layer)
model.add(layer1)

model.compile(
    loss=keras.losses.mse,
    optimizer=keras.optimizers.SGD(),
    metrics=keras.losses.mae
)
model.fit(array, array + 10, epochs=1)
