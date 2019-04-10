import tensorflow as tf
import numpy as np

# 5行
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
