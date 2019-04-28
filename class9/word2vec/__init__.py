# 文档集
documents = ["Human machine interface for lab abc computer applications A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
# 去冠词,连接词等
stopList = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stopList]
         for document in documents]
# 统计词频
from collections import defaultdict

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
# 去掉低频词
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
from pprint import pprint

pprint(texts)

# 获取词库dictionary
from gensim import corpora

dictionary = corpora.Dictionary(texts)
# dictionary.save('/tmp/deerwester.dict')
print(dictionary)
print(dictionary.token2id)

# 将文档转为语料库corpus
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
