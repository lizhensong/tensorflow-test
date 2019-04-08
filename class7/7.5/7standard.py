import matplotlib.pyplot as plt
import tensorflow as tf

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_jpeg(image)
    # tf 的图像标准化处理 将均值变为0，方差变成1
    standard = tf.image.per_image_standardization(img_after_decode)

    plt.imshow(standard.eval())
    plt.show()

