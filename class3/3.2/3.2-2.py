import tensorflow as tf

# 使用Graph()函数创建一个计算图
g1 = tf.Graph()
# 将定义的计算图使用as_default()函数设为默认
with g1.as_default():
    # 创建计算图中的变量并设置初始值为
    a = tf.get_variable("a", [2], initializer=tf.ones_initializer())
    b = tf.get_variable("b", [2], initializer=tf.zeros_initializer())

# 使用Graph()函数创建另一个计算图
g2 = tf.Graph()
with g2.as_default():
    a = tf.get_variable("a", [2], initializer=tf.zeros_initializer())
    b = tf.get_variable("b", [2], initializer=tf.ones_initializer())

# 这样就开启TensorFlow session
with tf.Session(graph=g1) as sess:
    # 初始化这个计算图中的所有变量
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("a")))
        print(sess.run(tf.get_variable("b")))
        # 打印[1. 1.]  [0. 0.]

# with/as是python的环境上下文管理器。一般Session是必须关闭资源的（防止资源泄漏）。
# 使用这个管理器会自动关闭所有不用的资源。同时其包含异常处理机制。

# 在定义会话的时候，可以使用tf.InteractiveSession()。
# 可以省去将产生的会话通过as_default()函数注册为默认会话的过程。
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("a")))
        print(sess.run(tf.get_variable("b")))
        # 打印[0. 0.]  [1. 1.]
