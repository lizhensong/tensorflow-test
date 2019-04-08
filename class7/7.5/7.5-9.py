import matplotlib.pyplot as plt
import tensorflow as tf

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型central_crop(image,central_fraction)
    # central_fraction在0到1之间（中心裁剪）
    central_cropped = tf.image.central_crop(img_after_decode, 0.4)
    plt.imshow(central_cropped.eval())
    plt.show()
