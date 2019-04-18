# 将Cifar-10的pickle文件转为tensorFlow的TFRecord文件，
# 方便使用tensorFlow处理
import tensorflow as tf
import pickle
import numpy


# 将值转换为TFRecord的要求格式。value接收的是一个列表。
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 传入一个字典。
# 这个字典中key是要保存的名字，value必须是rf.train.Feature的格式。
def tf_example(feature):
    # 定义一个features 将feature这个字典写入。
    features = tf.train.Features(feature=feature)
    # 定义一个Example，将相关信息写入到这个数据结构
    example = tf.train.Example(features=features)
    # 将一个Example写入到TFRecord文件
    # 原型writer(self, record)
    # example.SerializeToString()将example压缩为二进制数据。
    return example.SerializeToString()


def load(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def train_load():
    path_train = 'D:/Python_Work_Space/learning-data/cifar-10-batches-py/data_batch_'
    for i in range(1, 6):
        train = load(path_train + '{}'.format(i))
        images_1 = numpy.reshape(train[b'data'], (10000, 3, 32, 32))
        images = numpy.transpose(images_1, (0, 2, 3, 1))
        # 图片类别下标    softmax求解的时候传入的是下标所以没问题
        labels = train[b'labels']
        train_save = 'D:/Python_Work_Space/learning-data/cifar-10-batches-py/TFRecord/train-'
        with tf.python_io.TFRecordWriter(train_save + '{}'.format(i)) as writer:
            for j in range(10000):
                image_to_string = images[j].tostring()
                feature_info = {
                    "label": int64_feature(labels[j]),
                    "image": bytes_feature(image_to_string)
                }
                writer.write(tf_example(feature_info))


def cifar_test_load():
    path_test = 'D:/Python_Work_Space/learning-data/cifar-10-batches-py/test_batch'
    test = load(path_test)
    images_1 = numpy.reshape(test[b'data'], (10000, 3, 32, 32))
    images = numpy.transpose(images_1, (0, 2, 3, 1))
    # 图片类别下标    softmax求解的时候传入的是下标所以没问题
    labels = test[b'labels']
    test_save = 'D:/Python_Work_Space/learning-data/cifar-10-batches-py/TFRecord/test'
    with tf.python_io.TFRecordWriter(test_save) as writer:
        for i in range(10000):
            image_to_string = images[i].tostring()
            feature_info = {
                "label": int64_feature(labels[i]),
                "image": bytes_feature(image_to_string)
            }
            writer.write(tf_example(feature_info))


if __name__ == '__main__':
    train_load()
    cifar_test_load()
