import tensorflow as tf

# 声明两个变量并计算其加和
# 如果定义了name属性，调用会以name属性为主。name如果改变，在载入的时候需要指定变量名。
a = tf.Variable(tf.constant([1.0, 2.0], shape=[2]), name="a")
b = tf.Variable(tf.constant([3.0, 4.0], shape=[2]), name="b")
result = a + b

# 定义初始化全部变量的操作
init_op = tf.global_variables_initializer()
# 定义Saver类对象用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 模型保存到/home/jiangziyang/model路径下的model.ckpt文件，其中model是模型的名称
    saver.save(sess, "./save_test/model.lzs")
    # save函数的原型是
    # save(self,ses,save_path,global_step,latest_filename,meta_graph_suffix,
    #                                            write_meta_graph, write_state)
