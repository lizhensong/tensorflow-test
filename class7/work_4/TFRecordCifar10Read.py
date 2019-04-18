import tensorflow as tf


def parser(example):
    features = tf.parse_single_example(
        example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
    # 图片必须先转为uint8
    images = tf.decode_raw(features["image"], tf.uint8)
    labels = tf.cast(features["label"], tf.int32)
    return images, labels


def read_train(batch_size=512):
    files = tf.data.Dataset.list_files('D:/Python_Work_Space/learning-data/cifar-10-batches-py/TFRecord/train-*')
    data_set = tf.data.TFRecordDataset(files).map(parser).shuffle(1024).repeat().batch(batch_size)
    iterator = data_set.make_initializable_iterator()
    return iterator


# 返回值 images labels
def read_test(batch_size=512):
    files = tf.data.Dataset.list_files('D:/Python_Work_Space/learning-data/cifar-10-batches-py/TFRecord/test')
    data_set = tf.data.TFRecordDataset(files).map(parser).shuffle(1024).repeat().batch(batch_size)
    iterator = data_set.make_initializable_iterator()
    return iterator
