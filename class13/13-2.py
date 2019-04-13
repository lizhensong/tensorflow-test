import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_steps = 1000
learning_rate = 0.001
batch_size = 100
data_dir = "D:\Python_Work_Space\learning-data\MNIST\data"
log_dir = "./board/log1"
mnist = input_data.read_data_sets(data_dir, one_hot=True)


def variable_summaries(var):
    with tf.name_scope("summaries"):
        # 求解函数传递进来的var参数的平均值，并使用scaler()函数进行汇总
        # tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
        # 第一个参数input_tensor： 输入的待降维的tensor;
        # 第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
        # 第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;(默认为False)
        mean = tf.reduce_mean(var)
        # 函数scalar()原型为scalar(name,tensor,collections)
        # 其中参数name是展示在ＴensorBoard上的标签，tensor就是要汇总的数据
        tf.summary.scalar("mean", mean)

        # 汇总var数据的标准差值,并将标签设为stddev
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)

        # 汇总var数据的最大值
        tf.summary.scalar("max", tf.reduce_max(var))

        # 汇总var数据的最小值
        tf.summary.scalar("min", tf.reduce_min(var))

        # 使用histogram()将var数据汇总为直方图的形式
        # 函数原型histogram(name,values,collections)
        # 其中参数name是展示在ＴensorBoard上的标签，tensor就是要汇总的数据
        tf.summary.histogram("histogram", var)


def create_layer(input_tensor, input_num, output_num, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            # 创建权重参数，并调用variable_summaries()方法统计权重参数的最大、最小
            # 均值、标准差等信息
            weights = tf.Variable(tf.truncated_normal([input_num, output_num], stddev=0.1))
            variable_summaries(weights)

        with tf.name_scope("biases"):
            # 创建偏偏置参数，并调用variable_summaries()方法统计偏置参数的最大、最小
            # 均值、方差等信息
            biases = tf.Variable(tf.constant(0.1, shape=[output_num]))
            variable_summaries(biases)

        with tf.name_scope("Wx_add_b"):
            # 计算没有加入激活的线性变换的结果，并通过histogram()函数汇总为直方图数据
            pre_activate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram("pre_activations", pre_activate)

        # 计算激活后的线性变换的结果，并通过histogram()函数汇总为直方图数据
        activations = act(pre_activate, name="activation")
        tf.summary.histogram("activations", activations)

        return activations


x = tf.placeholder(tf.float32, [None, 784], name="x-input")
y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")

hidden_1 = create_layer(x, 784, 500, "layer_0")
# tf.identity是节点赋值操作，a=b这个是普通赋值，不会生成tensor的节点
y = create_layer(hidden_1, 500, 10, "layer_1", act=tf.identity)

with tf.name_scope("input_reshape"):
    # 图片显示输入矩阵为4维。几张，高，宽，通道。
    # reshape中最多有一个-1意思是可以自动计算。
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image("{}input{}".format(y_, y), image_shaped_input, 10)

# 计算交叉熵损失并汇总为标量数据
with tf.name_scope("cross_entropy"):
    cross = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy = tf.reduce_mean(cross)
    tf.summary.scalar("cross_entropy_scalar", cross_entropy)

with tf.name_scope("train"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 计算预测精度并汇总为标量数据
with tf.name_scope("accuracy"):
    # arg_max 返回最大值的下标，如果下标相同true，反之。
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy_scalar", accuracy)

# 使用merge_all()函数直接获取所有汇总操作
merged = tf.summary.merge_all()

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 训练过程，测试过程
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + "/test")

    # 测试过程的feed数据
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    for i in range(max_steps):

        # 运行测试过程并输出日志文件到log下的test目录下
        if i % 100 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=test_feed)
            test_writer.add_summary(summary, i)
            print("Accuracy at step %s,accuracy is: %s%%" % (i, acc * 100))

        # 产生训练数据，运行训练过程
        else:
            x_train, y_train = mnist.train.next_batch(batch_size=batch_size)
            if i % 100 == 50:  # Record execution stats
                # 计算运算时运行时间和内存空间
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_train, y_: y_train},
                                      options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, "step%03d" % i)
                train_writer.add_summary(summary, i)

                # 注意，这里保存模型不是为了后期使用，而是为了可视化降维后的嵌入向量
                saver.save(sess, log_dir + "/model.ckpt", i)

                print("Adding run metadata for", i)
            else:

                summary, _ = sess.run([merged, train_op], feed_dict={x: x_train, y_: y_train})
                train_writer.add_summary(summary, i)

    # 关闭ＦileWriter
    train_writer.close()
    test_writer.close()
