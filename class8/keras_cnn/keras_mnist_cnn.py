from tensorflow import keras
import numpy as np

from class6.keras_use.mnist_read import load_mnist

path = 'D:\Python_Work_Space\learning-data\MNIST\data'
x_train, y_train = load_mnist(path, 'train')
x_test, y_test = load_mnist(path, 't10k')

train_images = np.reshape(x_train, (60000, 28, 28, 1))
test_images = np.reshape(x_test, (10000, 28, 28, 1))
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, y_train, epochs=5)

model.save_weights('./checkpoints/my_checkpoint')

test_loss, test_acc = model.evaluate(test_images, y_test)
print(test_acc)
