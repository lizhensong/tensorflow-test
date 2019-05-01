import tensorflow as tf
from tensorflow import keras

# npz中存储的是句子和标签（句子是one-hot向量的最大值向标组成）
path = 'D:\Python_Work_Space\learning-data\imdb-keras\imdb.npz'
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(path, num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

# json中存储的是词和其对应的one-hot向标（比npz中的词小3）
json_path = 'D:\Python_Work_Space\learning-data\imdb-keras\imdb_word_index.json'
# A dictionary mapping words to an integer index
word_index = keras.datasets.imdb.get_word_index(json_path)

# The first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])  # ？是不再字典中返回的


print(decode_review(train_data[0]))

# pad_sequences将传入的句子列表转换为2维[句子数，句中词数]
# 长句子切断，短句子填充。
# maxlen参数需要句子长度，value填充的值，
# padding（post、pre）post填充在后边，pre填充在前边
# truncating（post、pre）post在后边裁剪，pre在前边裁剪
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
# 增加L2范数
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

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

results = model.evaluate(test_data, test_labels)
print(results)
