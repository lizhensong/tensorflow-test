import tensorflow as tf

# 声明两个变量并计算其加和
# 如果定义了name属性，调用会以name属性为主。name如果改变，在载入的时候需要指定变量名。
a = tf.Variable(tf.constant([1.0, 2.0], shape=[2]), name="a1")
b = tf.Variable(tf.constant([3.0, 4.0], shape=[2]), name="b1")
result = a + b

# 定义Saver类对象用于保存模型
saver = tf.train.Saver()

saver.export_meta_graph('a', as_text=True)
