import gensim
import os
import logging
import zipfile
import tensorflow as tf


class MySentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for f_name in os.listdir(self.dir_name):
            with zipfile.ZipFile(os.path.join(self.dir_name, f_name)) as f:
                yield tf.compat.as_str(f.read(f.namelist()[0])).split()


sentences = MySentences('D:/Python_Work_Space/learning-data/vector')  # a memory-friendly iterator
# for i in sentences:
#     print(i)
# 可以传入一个迭代对象，将自动多次调用完成训练
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，
# 是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。
# 默认是0即Negative Sampling。
model = gensim.models.Word2Vec(sentences, window=3, hs=1)

# 模型的预测
print('-----------------分割线----------------------------')

# 计算两个词向量的相似度
try:
    sim1 = model.similarity('women', 'man')
    sim2 = model.similarity('king', 'queen')
except KeyError:
    sim1 = 0
    sim2 = 0
print(u'women 和 man 的相似度为 ', sim1)
print(u'king 和 queen 的相似度为 ', sim2)

print('-----------------分割线---------------------------')
# 与某个词最相近的3个字的词
print(u'与computer最相近的3个字的词')
req_count = 5
for key in model.similar_by_word('computer', topn=100):
    if len(key[0]) == 3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break

print('-----------------分割线---------------------------')
# 计算某个词的相关列表
try:
    sim3 = model.most_similar('computer', topn=20)
    print('和 computer 与相关的词有：\n')
    for key in sim3:
        print(key[0], key[1])
except KeyError:
    print(' error')

print('-----------------分割线---------------------------')
# 找出不同类的词
sim4 = model.doesnt_match('women computer queen king'.split())
print('women computer queen king')
print(u'上述中不同类的名词', sim4)

print('-----------------分割线---------------------------')
# 保留模型，方便重用
model.save('test.model')

# 对应的加载方式
# model2 = gensim.models.Word2Vec.load('test.model')
# 以一种c语言可以解析的形式存储词向量
# model.save_word2vec_format(u"书评.model.bin", binary=True)
# 对应的加载方式
# model_3 =word2vec.Word2Vec.load_word2vec_format("text8.model.bin",binary=True)
