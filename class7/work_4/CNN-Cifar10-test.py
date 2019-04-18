import tensorflow as tf
import numpy as np
from class7.work_4.TFRecordCifar10Read import read_test
from class7.work_4.TFRecordCifar10Read import read_train

batch_size = 2048
test_data = read_test(batch_size)
# test_data = read_train(batch_size)
bitch_test_data = test_data.get_next()

saver = tf.train.import_meta_graph('./board/model.ckpt-18400.meta')
with tf.Session() as sess:
    test_data.initializer.run()
    saver.restore(sess, tf.train.latest_checkpoint("./board"))
    graph = tf.get_default_graph()
    # 定义Saver类对象用于保存模型，可以准确查看属性定义名和操作名
    # saver.export_meta_graph('a', as_text=True)
    x = graph.get_tensor_by_name('x-input:0')
    y_ = graph.get_tensor_by_name('y-input:0')
    #
    x_image, y_label = sess.run(bitch_test_data)
    x_image_reshape = np.reshape(x_image, [batch_size, 32, 32, 3])
    test_feed = {x: x_image_reshape, y_: y_label}
    #
    accuracy = graph.get_tensor_by_name('accuracy_result:0')
    #
    a = sess.run([accuracy], feed_dict=test_feed)
    #
    print(a)
