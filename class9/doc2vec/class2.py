import gensim
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class MySentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        self.label = 0
        for f_name in os.listdir(self.dir_name):
            with open(os.path.join(self.dir_name, f_name)) as f:
                while True:
                    line = f.readline()
                    if line:
                        yield TaggedDocument(gensim.utils.simple_preprocess(line), [self.label])
                    else:
                        break
                    self.label += 1


documents = MySentences('D:/Python_Work_Space/learning-data/test')
# for i in documents:
#     print(i)
# vector_size 特征向量维度
# epochs 语料库上的迭代次数
model = Doc2Vec(documents, vector_size=50, window=2, min_count=0, workers=4, epochs=40)

model.save('./doc2cev.model')
model = Doc2Vec.load('./doc2cev.model')
# infer_vector 推断某一句的特征向量
# 不接收字符串，接收单词列表
model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
