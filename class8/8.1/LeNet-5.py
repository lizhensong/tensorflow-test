import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\Python_Work_Space\learning-data\MNIST\data", one_hot=True)

batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.99
max_steps = 30000


def hidden_layer(input_tensor, regularize, avg_class, reuse):
    # 创建第一层是个卷积层，输入28*28*1，卷积核5*5*1(32个)，得到第一层为28*28*32（padding全0填充：使输入输出高宽完全相同）
    # variable_scope和get_variable
    # 1.如果variable中的reuse参数是False，get_variable为创建张量（默认reuse为False）；
    # *在同一个变量空间内，连续创建同一个张量会报错。
    # 2.如果variable中的reuse参数是True，get_variable为获取已有张量；
    # *在同一个变量空间内，如果没有创建张量，而去获取张量，会报错。
    with tf.variable_scope("layer1-convolution", reuse=reuse):
        layer1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 每个核有一个偏移量
        layer1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        layer1_convolution = tf.nn.conv2d(input_tensor, layer1_weights, strides=[1, 1, 1, 1], padding="SAME")
        layer1 = tf.nn.relu(tf.nn.bias_add(layer1_convolution, layer1_biases))
    # 创建第二层是个池化层，池化后的结果为14*14*32（padding全0填充：使高和宽可以完美匹配过滤器）
    with tf.name_scope("layer2-max_pool"):
        layer2 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 创建第三层是个卷积层，输入为14*14*32，卷积核为5*5*32（64个）得到结果为14*14*64。（padding全0填充：使输入输出高宽完全相同）
    with tf.variable_scope("layer3-convolution", reuse=reuse):
        layer3_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        layer3_convolution = tf.nn.conv2d(layer2, layer3_weights, strides=[1, 1, 1, 1], padding="SAME")
        layer3 = tf.nn.relu(tf.nn.bias_add(layer3_convolution, layer3_biases))
    # 创建第四层是个池化层，池化后结果为7*7*64（padding全0填充：使高和宽可以完美匹配过滤器）
    # 将池化后的结果，展开m*3136
    with tf.name_scope("layer4-max_pool"):
        layer4_pool = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # tf获取形状：
        # 1.tf.shape()返回是一个tensor。必须sess.run()才可以得到其值。（普通数组、列表和张量都可以使用）
        # 2.tf.get_shape()返回的是一个元组，as_list()可以变为列表进行操作。（不能放入sess.run中会报错）
        shape = layer4_pool.get_shape().as_list()
        layer4 = tf.reshape(layer4_pool, [shape[0], 3136])
    # 创建第五层是个全连接层
    with tf.variable_scope("layer5-full", reuse=reuse):
        layer5_weights = tf.get_variable("weight", [3136, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularize(layer5_weights))
        layer5_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        if avg_class is None:
            layer5 = tf.nn.relu(tf.matmul(layer4, layer5_weights) + layer5_biases)
        else:
            layer5 = tf.nn.relu(tf.matmul(layer4, avg_class.average(layer5_weights)) + avg_class.average(layer5_biases))

    # 创建第五层是个全连接层
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
# 将滑动平均用到训练数据上
variables_averages_op = variable_averages.apply(tf.trainable_variables())
average_y = hidden_layer(x, l2_regularize, variable_averages, reuse=True)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

# staircase参数为False，学习率连续衰减
# 当其为True时，学习率梯度下降
learning_rate = tf.train.exponential_decay(learning_rate,
                                           training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay, staircase=True)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)

train_op_all = tf.group(train_op, variables_averages_op)

correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(max_steps):
        if i % 1000 == 0:
            x_val, y_val = mnist.validation.next_batch(batch_size)
            x_val_reshaped = np.reshape(x_val, (batch_size, 28, 28, 1))

            validate_accuracy = sess.run(accuracy, feed_dict={x: x_val_reshaped, y_: y_val})
            print("After {} training step(s) ,validation accuracy"
                  "using average model is {}%".format(i, validate_accuracy * 100))

        x_train, y_train = mnist.train.next_batch(batch_size)
        x_train_reshaped = np.reshape(x_train, (batch_size, 28, 28, 1))

        sess.run(train_op_all, feed_dict={x: x_train_reshaped, y_: y_train})
