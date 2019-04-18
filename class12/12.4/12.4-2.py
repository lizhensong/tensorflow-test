import tensorflow as tf

with tf.Session() as sess:
    with open('./model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')  # 导入计算图
    # graph_def中保存了图结构，可以从中查看具体的节点
    # train_writer = tf.summary.FileWriter('', graph_def)
    # train_writer.close()
    # 使用tensorboard查看节点
    # 需要先复原变量
    print(sess.run('b:0'))
    # 1

    # 输入
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op,  feed_dict={input_x: 5, input_y: 5})
    print(ret)
