import tensorflow as tf
import numpy as np
from class6.keras_use.mnist_read import load_mnist

path = 'D:\Python_Work_Space\learning-data\MNIST\data'
x_train, y_train = load_mnist(path, 'train')
x_test, y_test = load_mnist(path, 't10k')

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=[784, ]),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('\nTest accuracy:', test_acc)

# model.evaluate(x_test, y_test)

predictions = model.predict(x_test)
print(np.argmax(predictions[0]))
