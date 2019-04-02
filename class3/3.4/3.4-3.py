import tensorflow as tf

a = tf.placeholder(tf.float32, shape=(2), name="input")
b = tf.placeholder(tf.float32, shape=(4, 2), name="input")
result = a+b

with tf.Session() as sess:
    print(sess.run(result, feed_dict={a: [1.0, 2.0], b: [[2.0, 4.0],
                                                         [5.0, 6.0],
                                                         [7.0, 8.0],
                                                         [9.0, 10.0]]
                                      }))
print(result)
# Tensor("add:0", shape=(4, 2), dtype=float32)
# [[ 3.  6.]
# [ 6.  8.]
# [ 8. 10.]
# [ 10. 12.]]


