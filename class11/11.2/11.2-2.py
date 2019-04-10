import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from class11.TFRecord_IO import out

mnist = input_data.read_data_sets("D:\Python_Work_Space\learning-data\MNIST\data",
                                  dtype=tf.uint8, one_hot=True)

images = mnist.test.images
labels = mnist.test.labels
pixels = images.shape[1]
num_examples = mnist.test.num_examples

# num_files定义总共写入多少个文件
num_files = 2

for i in range(num_files):
    # 将数据写入多个文件时，为区分这些文件可以添加后缀
    filename = ("D:\Python_Work_Space\learning-data\MNIST\data_tfrecords-{}-of-{}".format(i, num_files))
    writer = tf.python_io.TFRecordWriter(filename)

    # 将Example结构写入TFRecord文件，写入文件的过程和11.1节一样。
    for index in range(num_examples):
        image_string = images[index].tostring()
        feature = {
            "pixels": out.int64_feature(pixels),
            "label": out.int64_feature(np.argmax(labels[index])),
            "image_raw": out.bytes_feature(image_string)
        }
        writer.write(out.tf_example(feature))
    writer.close()
