import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\Python_Work_Space\learning-data\MNIST\data", one_hot=True)

batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.99
max_steps = 30000


def hidden_layer(input_tensor, regularize, avg_class, reuse):
    with tf.variable_scope("layer1-convolution", reuse=reuse):
        layer1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        layer1_convolution = tf.nn.conv2d(input_tensor, layer1_weights, strides=[1, 1, 1, 1], padding="SAME")
        layer1 = tf.nn.relu(tf.nn.bias_add(layer1_convolution, layer1_biases))
    with tf.name_scope("layer2-max_pool"):
        layer2 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    with tf.variable_scope("layer3-convolution", reuse=reuse):
        layer3_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        layer3_convolution = tf.nn.conv2d(layer2, layer3_weights, strides=[1, 1, 1, 1], padding="SAME")
        layer3 = tf.nn.relu(tf.nn.bias_add(layer3_convolution, layer3_biases))
    with tf.name_scope("layer4-max_pool"):
        layer4_pool = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape = layer4_pool.get_shape().as_list()
        layer4 = tf.reshape(layer4_pool, [shape[0], 3136])
    with tf.variable_scope("layer5-full", reuse=reuse):
        layer5_weights = tf.get_variable("weight", [3136, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularize(layer5_weights))
        layer5_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        if avg_class is None:
            layer5 = tf.nn.relu(tf.matmul(layer4, layer5_weights) + layer5_biases)
        else:
            layer5 = tf.nn.relu(tf.matmul(layer4, avg_class.average(layer5_weights)) + avg_class.average(layer5_biases))
    with tf.variable_scope("layer6-full", reuse=reuse):
        layer6_weights = tf.get_variable("weight", [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularize(layer6_weights))
        layer6_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))
        if avg_class is None:
            result = tf.matmul(layer5, layer6_weights) + layer6_biases
        else:
            result = tf.matmul(layer5, avg_class.average(layer6_weights)) + avg_class.average(layer6_biases)
    return result


x = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="x-input")
y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")

l2_regularize = tf.contrib.layers.l2_regularizer(0.0001)

y = hidden_layer(x, l2_regularize, avg_class=None, reuse=False)

training_step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(0.99, training_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
average_y = hidden_layer(x, l2_regularize, variable_averages, reuse=True)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.math.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

learning_rate = tf.train.exponential_decay(learning_rate,
                                           training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay, staircase=True)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)

train_op_all = tf.group(train_op, variables_averages_op)

correct_prediction = tf.equal(tf.math.argmax(average_y, 1), tf.math.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("loss", loss)
tf.summary.scalar("learning_rate", learning_rate)
tf.summary.scalar("accuracy", accuracy)
merged = tf.summary.merge_all()
# keep_checkpoint_every_n_hours=数字  每多长时间保存一次
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter('./board/log1/train', sess.graph)
    test_writer = tf.summary.FileWriter('./board/log1/test')
    for i in range(max_steps):
        if i % 1000 == 0:
            x_val, y_val = mnist.validation.next_batch(batch_size)
            x_val_reshaped = np.reshape(x_val, (batch_size, 28, 28, 1))

            summary, validate_accuracy = sess.run([merged, accuracy], feed_dict={x: x_val_reshaped, y_: y_val})
            test_writer.add_summary(summary, i)
            print("After {} training step(s) ,validation accuracy"
                  "using average model is {}%".format(i, validate_accuracy * 100))
            saver.save(sess, './board/log1/model.ckpt', i)
        x_train, y_train = mnist.train.next_batch(batch_size)
        x_train_reshaped = np.reshape(x_train, (batch_size, 28, 28, 1))
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_op_all], feed_dict={x: x_train_reshaped, y_: y_train},
                              options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, "step{}".format(i))
        train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
