import matplotlib.pyplot as plt
import tensorflow as tf

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型（左上角为原点）
    # crop_to_bounding_box(image,offset_height,offset_width,target_height,target_width)
    # pad_to_bounding_box(image,offset_height,offset_width,target_height,target_width)
    croped = tf.image.crop_to_bounding_box(img_after_decode, 100, 100, 30, 30)
    padded = tf.image.pad_to_bounding_box(img_after_decode, 100, 100, 1000, 1000)

    plt.imshow(croped.eval())
    plt.show()
    plt.imshow(padded.eval())
    plt.show()
