import tensorflow as tf
from tensorflow import keras

# npz中存储的是句子和标签（句子是one-hot向量的最大值向标组成）
path = 'D:\Python_Work_Space\learning-data\imdb-keras\imdb.npz'
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(path, num_words=10000)

# pad_sequences将传入的句子列表转换为2维[句子数，句中词数]
# 长句子切断，短句子填充。
# maxlen参数需要句子长度，value填充的值，
# padding（post、pre）post填充在后边，pre填充在前边
# truncating（post、pre）post在后边裁剪，pre在前边裁剪
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=256)

vocab_size = 10000

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

new_model = keras.models.load_model('D:\Python_Work_Space/tensorflow-test\class9/num1/my_model.h5')
new_model.summary()
new_model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
loss, acc = new_model.evaluate(test_data, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
