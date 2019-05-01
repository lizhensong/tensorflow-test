import tensorflow as tf
import numpy as np

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
# lstm_cell = tf.keras.layers.LSTMCell(units=128)
# 32是bitch_size

inputs = tf.placeholder(np.float32, shape=(32, 100))

# 通过zero_state得到一个全0的初始状态
h0 = lstm_cell.zero_state(32, np.float32)
# h0 = lstm_cell.get_initial_state(inputs, 32, np.float32)

out, h1 = lstm_cell.__call__(inputs, h0)
print(h1.h)
print(h1.c)
