import tensorflow as tf


def parser(example):
    # 使用parse_single_example()函数解析读取的样例。
    # 原型parse_single_example(serialized,features,name,example_names)
    features = tf.parse_single_example(
        example,
        features={
            # 可以使用FixedLenFeature类对属性进行解析，
            "image_raw": tf.FixedLenFeature([], tf.string),
            "pixels": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64)
        })

    # decode_raw()函数用于将字符串解析成图像对应的像素数组
    # 函数原型decode_raw(bytes,out_type,little_endian,name)
    images = tf.decode_raw(features["image_raw"], tf.uint8)
    # 使用cast()函数进行类型转换
    labels = tf.cast(features["label"], tf.int32)
    pixels = tf.cast(features["pixels"], tf.int32)
    return images, labels, pixels


files = tf.data.Dataset.list_files('D:\Python_Work_Space\learning-data\MNIST\data_tfrecords-*')
# repeat 放在 batch前面
# num_parallel_reads=32 TFRecordDataset和map可以加这个开启多线程
data_set = tf.data.TFRecordDataset(files).map(parser).shuffle(1000).repeat().batch(100)
# shuffle repeat map batch num_parallel_reads

# make_initializable_iterator 可以使用这个填充 需要进行初始化
iterator = data_set.make_one_shot_iterator()
a, b, c = iterator.get_next()

with tf.Session() as sess:
    print(files)
    for i in range(10):
        image, label, pixel = sess.run([a, b, c])
        print(label)
