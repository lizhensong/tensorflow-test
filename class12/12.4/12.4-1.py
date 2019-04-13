import tensorflow as tf

a = tf.Variable(tf.constant(1.0, shape=[1]), name="a")
b = tf.Variable(tf.constant(2.0, shape=[1]), name="b")
result = a + b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 导出主要记录了TensorFlow计算图上节点信息的GraphDef部分
    # 使用get_default_graph()函数获取默认的计算图
    graph_def = sess.graph_def

    # convert_variables_to_constants()函数表示用相同值的常量替换计算图中所有变量，
    # 原型convert_variables_to_constants(sess,input_graph_def,output_node_names,
    #                          variable_names_whitelist, variable_names_blacklist)
    # sess是会话，
    # input_graph_def是具有节点的GraphDef对象，
    # output_node_names是要保存的计算图中的计算节点的名称，通常为字符串列表的形式，
    # variable_names_whitelist要转换为常量的变量名称集合(默认情况下，所有变量都将被转换)，
    # variable_names_blacklist是要省略转换为常量的变量名的集合。
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

    # 将导出的模型存入.pb文件 GFile和FastGFile一样。
    with tf.gfile.GFile("./model/model.pb", "wb") as f:
        # SerializeToString()函数用于将获取到的数据取出存到一个string对象中，
        # 然后再以二进制流的方式将其写入到磁盘文件中
        f.write(output_graph_def.SerializeToString())
