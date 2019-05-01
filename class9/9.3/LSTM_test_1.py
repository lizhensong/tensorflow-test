import tensorflow as tf
import numpy as np


def get_a_cell():
    return tf.keras.layers.LSTMCell(units=128)
    # return tf.nn.rnn_cell.BasicLSTMCell(num_units=128)


# 创建三层RNN
cell = tf.keras.layers.StackedRNNCells([get_a_cell() for _ in range(3)])
# cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])
print(cell.state_size)
# 32是bitch_size
inputs = tf.placeholder(np.float32, shape=(32, 100))

# 通过zero_state得到一个全0的初始状态
# h0 = cell.zero_state(32, np.float32)
h0 = cell.get_initial_state(inputs, 32, np.float32)

# 第一个是LSTM最后的输出（h），第二个是LSTM所有的h和c
out, h1 = cell.__call__(inputs, h0)
print(h1)
