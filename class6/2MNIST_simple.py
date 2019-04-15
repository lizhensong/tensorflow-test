import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def hidden_layer(input_tensor, weight_1, bias_1, weight_2, bias_2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weight_1)+bias_1)
    return tf.matmul(layer1, weight_2) + bias_2


mnist = input_data.read_data_sets("D:\Python_Work_Space\learning-data\MNIST\data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

weight1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[500]))
weight2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
y = hidden_layer(x, weight1, bias1, weight2, bias2)
# 这个函数中labels接收一个一维的数组，长度为m，每个的取值为[0,n)(就是类别，也是每一维最大值的下标)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

training_step = tf.Variable(0, trainable=False)

batch_size = 512                # 设置每一轮训练的batch大小

regularizer = tf.contrib.layers.l2_regularizer(0.0001)        # 计算L2正则化损失函数
regularization = regularizer(weight1)+regularizer(weight2)  # 计算模型的正则化损失
loss = tf.reduce_mean(cross_entropy)+regularization           # 总损失

train_op = tf.train.AdamOptimizer().minimize(loss, global_step=training_step)
# train_op_all = tf.group(train_op, averages_op)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 准备验证数据
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # 准备测试数据
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(30000):
        xs, ys = mnist.train.next_batch(512)
        sess.run(train_op, feed_dict={x: xs, y_: ys})
        if i % 1000 == 0:
            # 计算滑动平均模型在验证数据上的结果。
            # 为了能得到百分数输出，需要将得到的validate_accuracy扩大100倍
            train_accuracy = sess.run(accuracy, feed_dict={x: xs, y_: ys})
            print("After {} trainging step(s) ,train accuracy"
                  "using average model is {}%".format(i, train_accuracy * 100))
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After {} trainging step(s) ,validation accuracy"
                  "using average model is {}%" .format(i, validate_accuracy * 100))

    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After 30000 trainging step(s) ,test accuracy using average"
          " model is {}%".format(test_accuracy * 100))
