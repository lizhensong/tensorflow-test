import gensim
import os
import logging


class MySentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for f_name in os.listdir(self.dir_name):
            print(self.dir_name)
            with open(os.path.join(self.dir_name, f_name)) as f:
                while True:
                    line = f.readline()
                    if line:
                        yield line.split()
                    else:
                        break


sentences = MySentences('D:/Python_Work_Space/learning-data/test')  # a memory-friendly iterator
# for i in sentences:
#     print(i)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 可以传入一个迭代对象，将自动多次调用完成训练
# iter=n 默认iter为5 对象将调用6次：一次构建模型，5次为模型训练
model = gensim.models.Word2Vec(sentences)

# 使用不同的东西构建和训练可以用如下：
# model = gensim.models.Word2Vec(iter=1)  # an empty model, no training yet
# model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
# model.train(other_sentences)  # can be a non-repeatable, 1-pass generator

# 保存、加载、继续训练模型（可以在线训练）
# model.save('/tmp/mymodel')
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')
# model.train(more_sentences)

# 加载由原始C工具创建的模型（文本格式）：
# model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
# 加载由原始C工具创建的模型（二进制格式）
# model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)

# 获取训练后某个词的特征向量
# model['computer']  # raw NumPy vector of a word
# array([-0.00449447, -0.00310097, 0.02421786, ...], dtype=float32)

# 词向量相似性比较
# model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# [('queen', 0.50882536)]
# model.doesnt_match("breakfast cereal dinner lunch";.split())
# 'cereal'
# model.similarity('woman', 'man')
# 0.73723527
