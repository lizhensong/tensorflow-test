import tensorflow as tf
import numpy as np
from class8.Inception_V3.DataRead import read_test

test_input, test_label = read_test()
test_input = np.reshape(test_input, [-1, 2048])
test_label = np.reshape(test_label, [-1])

saver = tf.train.import_meta_graph('./board/model.ckpt-400.meta')
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint("./board"))
    graph = tf.get_default_graph()
    # 定义Saver类对象用于保存模型，可以准确查看属性定义名和操作名
    # saver.export_meta_graph('a', as_text=True)
    x = graph.get_tensor_by_name('x-input:0')
    y_ = graph.get_tensor_by_name('y-input:0')

    test_feed = {x: test_input, y_: test_label}
    #
    accuracy = graph.get_tensor_by_name('accuracy_result:0')
    #
    a = sess.run([accuracy], feed_dict=test_feed)
    #
    print(a)
