import tensorflow as tf


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
