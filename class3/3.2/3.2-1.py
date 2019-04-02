import tensorflow as tf
# 常量是一种输出值永远固定的计算
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([3.0, 4.0], name="b")
# 常量相加的计算
result = a+b
# 通过graph属性可以获取张量所属计算图
# 输出True  True
print(a.graph is tf.get_default_graph())
print(b.graph is tf.get_default_graph())
# 输出为一个张量表示。操作（节点名称：本结点第几个输出）、维度、数据类型。
print(result)

# 定义会话可以输出真实结果
with tf.Session() as sess:
    # 初始化所有变量
    tf.initialize_all_variables().run()
    print(sess.run(result))
