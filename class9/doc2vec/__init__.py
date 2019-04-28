from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# enumerate 将一个可迭代的对象变成
# 向标和对象的组合迭代对象
c = enumerate(common_texts)
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

model.save('./doc2cev.model')
model = Doc2Vec.load('./doc2cev.model')
