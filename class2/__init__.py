# TensorFlow的测试
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([3.0, 4.0], name="b")
result = a+b
sess = tf.Session()
sess.run(result)
