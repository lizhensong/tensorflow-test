import tensorflow as tf
from tensorflow import keras

import os

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

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
# 增加L2范数
model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.01), activation='relu'))
model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01), activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

# 模型载入1
# checkpoint_path = "training_1/cp.ckpt"
# model.load_weights(checkpoint_path)
# loss,acc = model.evaluate(test_data, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# 模型载入2
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model.load_weights(latest)
# results = model.evaluate(test_data, test_labels)
# print(results)
# 模型载入3
# model.load_weights('./checkpoints/my_checkpoint')
# loss, acc = model.evaluate(test_data, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# 模型载入4
# 由于不需要重新定义模型所有见：my_test_3
