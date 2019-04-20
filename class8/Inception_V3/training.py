import tensorflow as tf
import numpy as np
from class8.Inception_V3 import DataRead

max_steps = 30001
batch_size = 1024


def hidden_layer(input_tensor, regularize):
    with tf.name_scope("final_training_ops_1"):
        weights_1 = tf.Variable(tf.truncated_normal([2048, 1024], stddev=0.001), name='weight1')
        tf.add_to_collection("losses", regularize(weights_1))
        biases_1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='bias1')
        layer_1 = tf.nn.relu(tf.matmul(input_tensor, weights_1) + biases_1)
    with tf.name_scope("final_training_ops_2"):
        weights_2 = tf.Variable(tf.truncated_normal([1024, 5], stddev=0.001), name='weight2')
        tf.add_to_collection("losses", regularize(weights_2))
        biases_2 = tf.Variable(tf.constant(0.1, shape=[5]), name='bias2')
        return tf.matmul(layer_1, weights_2) + biases_2


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 2048], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

    l2_regularize = tf.contrib.layers.l2_regularizer(0.001)
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

    train_data = DataRead.read_train(batch_size)
    bitch_train_data = train_data.get_next()

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_data.initializer.run()
        train_writer = tf.summary.FileWriter('./board/train', sess.graph)
        valid_input, valid_label = DataRead.read_valid()
        valid_input = np.reshape(valid_input, [-1, 2048])
        valid_label = np.reshape(valid_label, [-1])
        for i in range(max_steps):
            x_image, y_label = sess.run(bitch_train_data)
            summary, _ = sess.run([merged, train_op],
                                  feed_dict={x: x_image, y_: y_label},
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, "step{}".format(i))
            train_writer.add_summary(summary, i)
            if i % 100 == 0:
                valid_accuracy = sess.run(accuracy, feed_dict={x: valid_input, y_: valid_label})
                print("After {} training step(s) ,validation is {}%".format(i, valid_accuracy * 100))
                if c_accuracy <= valid_accuracy:
                    saver.save(sess, './board/model.ckpt', i)
                    c_accuracy = valid_accuracy
        train_writer.close()
