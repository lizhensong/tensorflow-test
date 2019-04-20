import tensorflow as tf
import pickle


def parser(example):
    features = tf.parse_single_example(
        example,
        features={
            "image_run": tf.VarLenFeature(tf.float32),
            "label": tf.FixedLenFeature([], tf.int64)
        })
    return tf.sparse_tensor_to_dense(features["image_run"], default_value=0), features["label"]


def read_train(batch_size=2048):
    files = tf.data.Dataset.list_files('D:/Python_Work_Space/learning-data/flower_photos/data/tfRecord-training')
    data_set = tf.data.TFRecordDataset(files).map(parser).shuffle(4096).repeat().batch(batch_size)
    iterator = data_set.make_initializable_iterator()
    return iterator


def read_valid():
    path = 'D:/Python_Work_Space/learning-data/flower_photos/data/pickle-validation'
    with open(path, 'rb') as fo:
        valid_data = pickle.load(fo, encoding='bytes')
    return valid_data['输入'], valid_data['标签']


def read_test():
    path = 'D:/Python_Work_Space/learning-data/flower_photos/data/pickle-testing'
    with open(path, 'rb') as fo:
        test_data = pickle.load(fo, encoding='bytes')
    return test_data['输入'], test_data['标签']
