import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from class11.TFRecord_IO import out
# 获取的数据默认为float32,one_hot指定数据与类型的对应。
mnist = input_data.read_data_sets("D:\Python_Work_Space\learning-data\MNIST\data",
                                  dtype=tf.uint8, one_hot=True)

# 读取mnist数据。
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出TFRecord文件的地址(相对于系统根目录)。
filename = "D:\Python_Work_Space\learning-data\MNIST\MNIST_tfrecords"

# 创建一个python_io.TFRecordWriter()类的实例
with tf.python_io.TFRecordWriter(filename) as writer:

    # for循环执行了将数据填入到Example协议内存块的主要操作
    for i in range(num_examples):
        # 将图像矩阵转化成一个字符串
        image_to_string = images[i].tostring()
        # 这个字典中key是要保存的名字，value必须是rf.train.Feature的格式。
        feature = {
            "pixels": out.int64_feature(pixels),
            "label": out.int64_feature(np.argmax(labels[i])),
            "image_raw": out.bytes_feature(image_to_string)
        }

        writer.write(out.tf_example(feature))
