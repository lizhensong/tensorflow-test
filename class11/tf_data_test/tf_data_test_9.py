# tf.data.TextLineDataset()：
# 这个函数的输入是一个文件的列表，输出是一个dataset。dataset中的每一个元素就对应了文件中的一行。可以使用这个函数来读入CSV文件。
# tf.data.FixedLengthRecordDataset()：
# 这个函数的输入是一个文件的列表和一个record_bytes，之后dataset的每一个元素就是文件中固定字节数record_bytes的内容。通常用来读取以二进制形式保存的文件，如CIFAR10数据集就是这种形式。
# tf.data.TFRecordDataset()：
# 顾名思义，这个函数是用来读TFRecord文件的，dataset中的每一个元素就是一个TFExample。
import tensorflow as tf

limit = tf.placeholder(dtype=tf.int32, shape=[])

dataSet = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))

iterator = dataSet.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={limit: 10})
    for i in range(10):
        value = sess.run(next_element)
        print(value)
        assert i == value
