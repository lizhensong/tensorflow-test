import tensorflow as tf
import numpy as np

# shuffle 将分割的batch随机打乱 buffer_size最小打乱单元 一般使用1000（每1000个打乱）  数据比这个小的时候就不打乱
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    }).shuffle(buffer_size=5)

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")