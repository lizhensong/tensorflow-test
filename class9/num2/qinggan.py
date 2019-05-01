import tensorflow as tf
from tensorflow import keras

# npz中存储的是句子和标签（句子是one-hot向量的最大值向标组成）
path = 'D:\Python_Work_Space\learning-data\imdb-keras\imdb.npz'
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(path, num_words=10000)

vocab_size = 10000
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    # bidirectional将这一个生成时间序，return_sequences为True时每个时间点都有返回。
    # 没有时每个时间序在最后才有返回
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
# verbose 显示模式3种（0，无声；1，进度条；2，每一轮训练一行）
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

model.save_weights('./checkpoints/my_checkpoint')

results = model.evaluate(test_data, test_labels)
print(results)
