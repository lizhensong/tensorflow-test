import tensorflow as tf
import numpy as np
from class7.work_4.TFRecordCifar10Read import read_train

max_steps = 30001
batch_size = 1024


def hidden_layer(input_tensor, regularize):
    with tf.variable_scope("layer1-convolution"):
        layer1_weights = tf.get_variable("weight", [5, 5, 3, 32],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        layer1_convolution = tf.nn.conv2d(input_tensor, layer1_weights, strides=[1, 1, 1, 1], padding="SAME")
        layer1 = tf.nn.relu(tf.nn.bias_add(layer1_convolution, layer1_biases))
    with tf.name_scope("layer2-max_pool"):
        layer2 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    with tf.variable_scope("layer3-convolution"):
        layer3_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        layer3_convolution = tf.nn.conv2d(layer2, layer3_weights, strides=[1, 1, 1, 1], padding="SAME")
        layer3 = tf.nn.relu(tf.nn.bias_add(layer3_convolution, layer3_biases))
    with tf.name_scope("layer4-max_pool"):
        layer4_pool = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        layer4 = tf.reshape(layer4_pool, [-1, 4096])
    with tf.variable_scope("layer5-full"):
        layer5_weights = tf.get_variable("weight", [4096, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularize(layer5_weights))
        layer5_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        layer5 = tf.nn.relu(tf.matmul(layer4, layer5_weights) + layer5_biases)
    with tf.variable_scope("layer6-full"):
        layer6_weights = tf.get_variable("weight", [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection("losses", regularize(layer6_weights))
        layer6_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))
        layer6 = tf.matmul(layer5, layer6_weights) + layer6_biases
    return layer6


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x-input")
    y_ = tf.placeholder(tf.int64, [None], name="y-input")

    l2_regularize = tf.contrib.layers.l2_regularizer(0.0001)

    y = hidden_layer(x, l2_regularize)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection("losses"))

    training_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer().minimize(loss, global_step=training_step)

    correct_prediction = tf.equal(tf.math.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_result')

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    c_accuracy = 0

    train_data = read_train(batch_size)
    bitch_train_data = train_data.get_next()

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_data.initializer.run()
        train_writer = tf.summary.FileWriter('./board/train', sess.graph)
        for i in range(max_steps):
            x_image, y_label = sess.run(bitch_train_data)
            x_image_reshape = np.reshape(x_image, [batch_size, 32, 32, 3])

            if i % 100 == 0:
                validate_accuracy, summary, _ = sess.run([accuracy, merged, train_op],
                                                         feed_dict={x: x_image_reshape, y_: y_label},
                                                         options=run_options, run_metadata=run_metadata)
                print("After {} training step(s) ,validation accuracy using average model is {}%"
                      .format(i, validate_accuracy * 100))

                if c_accuracy <= validate_accuracy:
                    saver.save(sess, './board/model.ckpt', i)
                    c_accuracy = validate_accuracy
            else:
                summary, _ = sess.run([merged, train_op],
                                      feed_dict={x: x_image_reshape, y_: y_label},
                                      options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, "step{}".format(i))
            train_writer.add_summary(summary, i)
        train_writer.close()
