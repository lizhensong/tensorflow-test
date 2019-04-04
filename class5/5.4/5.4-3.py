import tensorflow as tf

# 定义x作为需要进行dropout处理的数据
x = tf.Variable(tf.ones([10, 10]))

# 定义dro作为dropout处理时代keep_prob参数
dro = tf.placeholder(tf.float32)

# 定义一个dropout操作
# 数原型dropout(x,keep_prob,noise_shape,seed,name)
y = tf.nn.dropout(x, rate=1-dro)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(x))
    print(sess.run(y, feed_dict={dro: 0.5}))

'''print输出的结果
    [[2. 0. 2. 0. 0. 0. 2. 2. 0. 2.]
     [0. 0. 2. 0. 0. 2. 0. 2. 0. 2.]
     [0. 0. 0. 2. 2. 2. 2. 2. 2. 0.]
     [2. 0. 0. 2. 2. 2. 2. 2. 0. 0.]
     [2. 2. 2. 0. 2. 0. 2. 2. 2. 2.]
     [0. 2. 0. 0. 2. 0. 2. 0. 0. 2.]
     [0. 2. 2. 2. 2. 2. 0. 0. 2. 2.]
     [2. 0. 0. 2. 2. 0. 2. 0. 2. 2.]
     [2. 0. 2. 2. 2. 0. 0. 2. 2. 0.]
     [0. 2. 0. 0. 0. 2. 0. 2. 2. 0.]]
     '''