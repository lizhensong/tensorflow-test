import tensorflow as tf
# 产生一个给定值的常量
a1 = tf.get_variable("a1", [1], initializer=tf.constant_initializer(1.0))
a2 = tf.Variable(tf.constant(1.0, shape=[1], name='a2'))
# constant()给定常量
# random_normal()满足正态分布的随机数值
# ...
