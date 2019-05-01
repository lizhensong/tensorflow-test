from numpy import array
import tensorflow as tf

# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape((len(seq), 1, 1))
y = seq.reshape((len(seq), 1))
# define LSTM configuration
n_neurons = length
n_batch = length
n_epoch = 1000
# create LSTM
x_input = tf.placeholder(tf.float32, shape=(None, 100))
labels = tf.placeholder(tf.float32, shape=(None, 10))

LSTM_layer = tf.keras.layers.LSTM(n_neurons)(x_input)
Dense_layer = tf.keras.layers.Dense(10, activation='softmax')(LSTM_layer)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dense_layer, labels=labels))

train_optim = tf.train.AdamOptimizer().minimize(loss)

acc_pred = tf.keras.metrics.categorical_accuracy(labels, Dense_layer)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for _ in range(n_epoch):
       pass
