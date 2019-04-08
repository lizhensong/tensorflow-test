import matplotlib.pyplot as plt
import tensorflow as tf

with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    img_after_decode = tf.image.decode_png(image)

    # 函数原型expand_dims(input,axis,name,dim) 给图像添加一个维度。第一个位置，张数。
    # 用tf.cast(, )这个类型转换在后边输出的时候还有转回来。并且框将去除颜色变成黑色。
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_after_decode, tf.float32), 0)

    # 定义边框的坐标系数
    boxes = tf.constant([[[0.1, 0.05, 0.9, 0.9], [0.4, 0.4, 0.6, 0.6]]])

    # 绘制边框，函数原型draw_bounding_boxes(images,boxes,name)
    image_boxed = tf.image.draw_bounding_boxes(batched, boxes)

    # draw_bounding_boxes()函数处理的是一个batch的图片，如果此处给imshow()函数
    # 传入image_boxed参数会造成报错(Invalid dimensions for image data)
    plt.imshow(image_boxed[0].eval())
    plt.show()
