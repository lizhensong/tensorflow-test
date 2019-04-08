import matplotlib.pyplot as plt
import tensorflow as tf

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型resize_image_with_crop_or_pad(image,target_height,target_width)
    # 裁剪图像 （中心裁剪）
    croped = tf.image.resize_image_with_crop_or_pad(img_after_decode, 30, 30)
    # 填充图像
    padded = tf.image.resize_image_with_crop_or_pad(img_after_decode, 1000, 1000)

    # 用pyplot显示结果
    plt.imshow(croped.eval())
    plt.show()
    plt.imshow(padded.eval())
    plt.show()
