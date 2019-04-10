import tensorflow as tf
import numpy as np

# repeat 数据集的重复次数，读入一次是1  2表示将读俩次
# 如果直接调用repeat()没有参数  会无限的读下去    不会报错
dataSet = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    }).repeat(2)

iterator = dataSet.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
