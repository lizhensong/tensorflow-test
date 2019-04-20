import tensorflow as tf
import numpy as np
import pickle
from class8.Inception_V3.picture_class_op import getfilename
from class11.TFRecord_IO import out

with tf.Session() as sess:
    with open('./model/tensorflow_inception_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')  # 导入计算图
        # train_writer = tf.summary.FileWriter('./model/board', graph_def)
        # train_writer.close()
        input_image = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')
        output_tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')
    path = 'D:/Python_Work_Space/learning-data/flower_photos/photo'
    file_name = getfilename(path)
    # 处理training集合
    with tf.python_io.TFRecordWriter(
            'D:/Python_Work_Space/learning-data/flower_photos/data/tfRecord-training') as writer:
        k = 0
        for i in file_name['training']:
            for j in file_name['training'][i]:
                p_path = path + '/' + i + '/' + j
                with open(p_path, 'rb')as fp:
                    image = fp.read()
                output_num = sess.run(output_tensor, feed_dict={input_image: image})
                feature = {
                    "image_run": out.float_feature(np.squeeze(output_num)),
                    "label": out.int64_feature([k])
                }
                writer.write(out.tf_example(feature))
                print('处理完{}'.format(j))
            k += 1
            print('哈哈处理完{}'.format(i))
    # 处理valid集
    k = 0
    image_run = []
    label = []
    for i in file_name['validation']:
        for j in file_name['validation'][i]:
            p_path = path + '/' + i + '/' + j
            with open(p_path, 'rb')as fp:
                image = fp.read()
            output_num = sess.run(output_tensor, feed_dict={input_image: image})
            image_run.append(np.squeeze(output_num))
            label.append(k)
            print('处理完{}'.format(j))
        k += 1
        print('哈哈处理完{}'.format(i))
    with open('D:/Python_Work_Space/learning-data/flower_photos/data/pickle-validation', 'wb') as f:
        pickle.dump({'输入': image_run, '标签': label}, f)
    print('写入完成')
    # 处理test集
    k = 0
    image_run = []
    label = []
    for i in file_name['testing']:
        for j in file_name['testing'][i]:
            p_path = path + '/' + i + '/' + j
            with open(p_path, 'rb')as fp:
                image = fp.read()
            output_num = sess.run(output_tensor, feed_dict={input_image: image})
            image_run.append(np.squeeze(output_num))
            label.append(k)
            print('处理完{}'.format(j))
        k += 1
        print('哈哈处理完{}'.format(i))
    with open('D:/Python_Work_Space/learning-data/flower_photos/data/pickle-testing', 'wb') as f:
        pickle.dump({'输入': image_run, '标签': label}, f)
    print('写入完成')
