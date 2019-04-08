import matplotlib.pyplot as plt
import tensorflow as tf

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型adjust_hue(image,delta,name)
    adjusted_hue1 = tf.image.adjust_hue(img_after_decode, 0.1)
    adjusted_hue2 = tf.image.adjust_hue(img_after_decode, 0.3)
    adjusted_hue3 = tf.image.adjust_hue(img_after_decode, 0.6)
    adjusted_hue4 = tf.image.adjust_hue(img_after_decode, 0.9)

    plt.imshow(adjusted_hue2.eval())
    plt.show()

    # random_hue()函数原型为random_hue(image, max_delta)
    # 功能是在[-max_delta, max_delta]的范围随机调整图片的色相。
    # max_delta的取值在[0, 0.5]之间。
