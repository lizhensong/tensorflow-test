import matplotlib.pyplot as plt
import tensorflow as tf

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)
    boxes = tf.constant([[[0.1, 0.05, 0.9, 0.9], [0.4, 0.4, 0.6, 0.6]]])

    # 函数原型
    # sample_distorted_bounding_box(image_size,bounding_boxes,seed,seed2,min_object_covered,
    #       aspect_ratio_range,area_range,max_attempts,use_image_if_no_bounding_boxes,name)
    # tf.constant的返回可以作为这个函数bounding_boxes这个参数的传入。第一个参数是图像的形状。
    begin, size, bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(img_after_decode), bounding_boxes=boxes)

    batched = tf.expand_dims(tf.image.convert_image_dtype(img_after_decode, tf.float32), 0)

    image_boxed = tf.image.draw_bounding_boxes(batched, bounding_box)

    # slice()函数原型slice(input_,begin,size,name)
    # 随机裁剪。
    sliced_image = tf.slice(img_after_decode, begin, size)

    image_boxed_all = tf.image.draw_bounding_boxes(batched, boxes)

    plt.imshow(image_boxed[0].eval())
    plt.show()
    plt.imshow(sliced_image.eval())
    plt.show()
    plt.imshow(image_boxed_all[0].eval())
    plt.show()
