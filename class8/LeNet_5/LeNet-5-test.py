import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\Python_Work_Space\learning-data\MNIST\data", one_hot=True)

saver = tf.train.import_meta_graph('./board/log1/model.ckpt-29000.meta')
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint("./board/log1"))
    graph = tf.get_default_graph()
    # 定义Saver类对象用于保存模型，可以准确查看属性定义名和操作名
    # saver = tf.train.Saver()
    # saver.export_meta_graph('a', as_text=True)
    x = graph.get_tensor_by_name('x-input:0')
    y_ = graph.get_tensor_by_name('y-input:0')

    x_val, y_val = mnist.test.next_batch(100)
    x_val_reshaped = np.reshape(x_val, (100, 28, 28, 1))
    test_feed = {x: x_val_reshaped, y_: y_val}

    ArgMax_1 = graph.get_tensor_by_name('ArgMax_1:0')
    ArgMax_2 = graph.get_tensor_by_name('ArgMax_2:0')
    Mean_1 = graph.get_tensor_by_name('Mean_1:0')

    a, b, c = sess.run([ArgMax_1, ArgMax_2, Mean_1], feed_dict=test_feed)

    print(a)
    print(b)
    print(c)
