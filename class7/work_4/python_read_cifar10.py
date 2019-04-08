# python cifar10的简单读取方法。
import pickle
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf


def load(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


p = 'D:\Python_Work_Space\learning-data\cifar-10-batches-py\data_batch_1'
d = load(p)
# 图片数，深度（通道数），行数（高），列数（宽）。
images = numpy.reshape(d[b'data'], (10000, 3, 32, 32))
with tf.Session() as sess:
    a = tf.transpose(images, [0, 2, 3, 1])
    print(a)
    for i in range(0, 10000):
        # 有输出的可以用eval
        plt.imshow(a.eval()[i])
        plt.show()
