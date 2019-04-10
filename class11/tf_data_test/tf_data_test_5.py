import tensorflow as tf
import numpy as np

# batch 将分割的数据，合成n个一组。（数据分割成为batch是使用）
dataset = tf.data.Dataset.from_tensor_slices({
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    }).batch(2)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

# {'a': array([1., 2.]), 'b': array([[0.38629342, 0.26578067],
#        [0.47984543, 0.92009353]])}
# {'a': array([3., 4.]), 'b': array([[0.90793437, 0.49425597],
#        [0.80169309, 0.57492318]])}
# {'a': array([5.]), 'b': array([[0.07918089, 0.69103132]])}
# end!
