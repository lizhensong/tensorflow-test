import matplotlib.pyplot as plt
import tensorflow as tf

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_jpeg(image)

    # 函数原型adjust_contrast(images,contrast_factor)
    # 将图片的饱和度调整-6。
    adjusted_saturation = tf.image.adjust_saturation(img_after_decode, -6)
    # 将图片的饱和度调整+6。
    # adjusted_saturation = tf.image.adjust_saturation(img_after_decode, 6)
    plt.imshow(adjusted_saturation.eval())
    plt.show()

    # random_saturation()函数用于在[lower, upper]的范围随机调整图的饱和度。
    # 函数原型为random_saturation(image, lower, upper,seed)
