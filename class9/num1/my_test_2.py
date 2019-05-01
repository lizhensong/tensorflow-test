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

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

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

# 模型保存1
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# # Create checkpoint callback
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
# model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512,
#           validation_data=(x_val, y_val),
#           callbacks=[cp_callback])
# 保存模型2
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, verbose=1, save_weights_only=True,
#     # Save weights, every 5-epochs.
#     period=5)
# model.save_weights(checkpoint_path.format(epoch=0))
#
# model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=512,
#           validation_data=(x_val, y_val),
#           callbacks=[cp_callback])
# 保存模型3
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=10,
#                     batch_size=512,
#                     validation_data=(x_val, y_val),
#                     verbose=1)
# model.save_weights('./checkpoints/my_checkpoint')
# 保存模型4
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
model.save('my_model.h5')
