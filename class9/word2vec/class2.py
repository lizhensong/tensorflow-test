import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]
# min_count忽略总频数小于1的所有词
# min_count默认为5
# min_count 的合理值介于0到100之间，具体取决于数据集的大小
# size 特征向量的维度：默认值为100
# 合理的价值在几十到几百
# workers 并行运算的线程数 默认为3
# 需要安装Cython才可以启动这个设置
model = gensim.models.Word2Vec(sentences, min_count=1)

print(model)

# 谷歌发布了大约20,000个句法和语义测试示例的测试集，遵循“A is to B as C is to D”任务：
# https：//raw.githubusercontent.com/RaRe-Technologies/gensim/develop/gensim/ test / test_data / questions-words.txt。
# 这个函数可以评估模型生成特征向量的准确性
# model.accuracy('/tmp/questions-words.txt')
# 2014-02-01 22:14:28,387 : INFO : family: 88.9% (304/342)
# 2014-02-01 22:29:24,006 : INFO : gram1-adjective-to-adverb: 32.4% (263/812)
# 2014-02-01 22:36:26,528 : INFO : gram2-opposite: 50.3% (191/380)
# 2014-02-01 23:00:52,406 : INFO : gram3-comparative: 91.7% (1222/1332)
# 2014-02-01 23:13:48,243 : INFO : gram4-superlative: 87.9% (617/702)
# 2014-02-01 23:29:52,268 : INFO : gram5-present-participle: 79.4% (691/870)
# 2014-02-01 23:57:04,965 : INFO : gram7-past-tense: 67.1% (995/1482)
# 2014-02-02 00:15:18,525 : INFO : gram8-plural: 89.6% (889/992)
# 2014-02-02 00:28:18,140 : INFO : gram9-plural-verbs: 68.7% (482/702)
# 2014-02-02 00:28:18,140 : INFO : total: 74.3% (5654/7614)
