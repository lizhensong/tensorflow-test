import tensorflow as tf
import numpy as np

cell = tf.keras.layers.LSTMCell(units=128)
print(cell.state_size)
# 32是bitch_size
inputs = tf.placeholder(np.float32, shape=(32, 10, 100))

h0 = cell.get_initial_state(inputs, 32, np.float32)
layer = tf.keras.layers.RNN(cell)
y = layer(inputs)
print(y)
