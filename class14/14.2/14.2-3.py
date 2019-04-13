import tensorflow as tf

# 通过device()函数将运算指定到GPU设备上。
# 注意这里a和b不再是浮点值，而是整数
with tf.device("/gpu:0"):
    a = tf.Variable(tf.constant([1, 2], shape=[2]), name="a")
    b = tf.Variable(tf.constant([3, 4], shape=[2]), name="b")
result = a + b
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    print(sess.run(result))
