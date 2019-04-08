import matplotlib.pyplot as plt
import tensorflow as tf
# import numpy as np

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型resize_images(images,size,method,align_corners)
    resize = tf.image.resize_images(img_after_decode, [1000, 1000], method=3)

    print(resize.dtype)
    # 打印的信息<dtype: 'float32'>

    # 从print的结果看出经由resize_images()函数处理图片后返回的数据是float32格式的，
    # 所以需要转换成uint8才能正确打印图片，这里使用np.asarray()存储了转换的结果
    # resize = np.asarray(resize.eval(), dtype="uint8")

    # 使用tf中的cast函数转换类型
    resize = tf.cast(resize, tf.uint8)

    plt.imshow(resize.eval())
    plt.show()
