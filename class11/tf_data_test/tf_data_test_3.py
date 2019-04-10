import tensorflow as tf
import numpy as np

# 字典  不同类型集合必须各自所含数据个数相同。
dataset = tf.data.Dataset.from_tensor_slices({
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    })
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")

# {'a': 1.0, 'b': array([0.46376747, 0.79287722])}
# {'a': 2.0, 'b': array([0.8429679 , 0.93711924])}
# {'a': 3.0, 'b': array([0.38351752, 0.61388968])}
# {'a': 4.0, 'b': array([0.43687297, 0.87793769])}
# {'a': 5.0, 'b': array([0.91385896, 0.7366697 ])}
# end!
