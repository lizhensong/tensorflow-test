import tensorflow as tf
from tensorflow.python.framework import graph_util


with tf.Session() as sess:
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    # 这里的输出需要加上name属性
    op = tf.add(xy, b, name='op_to_store')

    sess.run(tf.global_variables_initializer())

    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

    # 测试 OP
    feed_dict = {x: 10, y: 3}
    print(sess.run(op, feed_dict))

    # 写入序列化的 PB 文件
    with open('./model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
