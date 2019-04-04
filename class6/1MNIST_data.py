from tensorflow.examples.tutorials.mnist import input_data
# 获取杨教授数字识别数据，将数据分为训练集、验证集、测试集。
# one_hot 是否将样本图片对应到标注信息（label分类）
mnist = input_data.read_data_sets("D:\Python_Work_Space\MNIST\data", one_hot=True)
print("训练集：")
print(mnist.train.images.shape, mnist.train.labels.shape)
print("验证集：")
print(mnist.validation.images.shape, mnist.validation.labels.shape)
print("测试集：")
print(mnist.test.images.shape, mnist.test.labels.shape)
