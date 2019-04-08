import matplotlib.pyplot as plt
import tensorflow as tf

# 读取原始的图像
with tf.gfile.GFile("./test/dog.png", 'rb') as f1:
    image = f1.read()
with tf.Session() as sess:
    # TensorFlow提供了decode_png()函数将.png格式的图像解码从而得到图像对应的三位矩阵
    # 函数原型decode_png(contents,channels,dtype,name)
    img_after_decode = tf.image.decode_png(image)

    # 输出解码之后的三维矩阵，并调用pyplot工具可视化得到的图像
    print(img_after_decode.eval())
    a = tf.transpose(img_after_decode, [1, 0, 2])
    plt.imshow(a.eval())
    plt.show()

    # 这一句是为了方便后续的样例程序对图像进行处理
    # img_after_decode = tf.image.convert_image_dtype(img_after_decode,dtype = tf.float32)

    # TensorFlow提供了encode_png()函数将解码后的图像进行再编码
    # 函数原型encode_png(image,compression,name)
    encode_image = tf.image.encode_png(img_after_decode)
    with tf.gfile.GFile("./test/dog1.png", "wb") as f2:
        f2.write(encode_image.eval())

    # decode_jpeg()函数用于解码.jpeg/.jpg格式的图像，原型
    # decode_jpeg(contents,channels,ratio,fancy_upscaling,try_recover_truncated,
    #                                           acceptable_fraction,dct_method,name)
    # decode_gif()函数用于解码.gif格式的图像，原型
    # decode_gif(contents,name)

    # encode_jpeg()函数用于编码为.jpeg/.jpg格式的图像，原型
    # encode_jpeg(image,format,quality,progressive,optimize_size,chroma_downsampling,
    #                                density_unit,x_density,y_density,xmp_metadata,name)
