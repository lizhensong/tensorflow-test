import numpy as np
import tensorflow as tf

# 从硬盘中读入两个Numpy数组
with np.load("/var/data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})
