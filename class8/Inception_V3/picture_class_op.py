import os


def getfilename(path):
    path_class = {'training': {}, 'validation': {}, 'testing': {}}
    f_list = os.listdir(path)
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '':
            for key in path_class:
                path_class[key][i] = []
            path_document = path + '/' + i
            p_list = os.listdir(path_document)
            for j in p_list:
                if os.path.splitext(j)[1] != '.jpg':
                    p_list.remove(j)
            train_len = int(len(p_list)*0.6)
            valid_len = int(len(p_list)*0.8)
            path_class['training'][i].extend(p_list[:train_len])
            path_class['validation'][i].extend(p_list[train_len:valid_len])
            path_class['testing'][i].extend(p_list[valid_len:])
    return path_class
