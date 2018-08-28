snownlp源码分析（情感分析部分）
==========
snownlp github:https://github.com/isnowfy/snownlp  

调用snownlp的sentiments方法，代码如下：

```python
from snownlp import SnowNLP
 
#创建snownlp对象，设置要测试的语句
s = SnowNLP('这东西不错。。')
# 调用sentiments方法获取积极情感概率
print(s.sentiments)
```
输出：0.8371034573341097  
snownlp的情感分析中，输出在0~1之间，认为大于0.5是积极，小于0.5是消极。
## snownlp包功能
* classification
   * bayes.py
* normal
   * \__init\__.py
   * pinyin.py
   * pinyin.txt
   * stopwords.txt
   * zh.py
* seg
* sentiment
   * \__init\__.py
   * neg.txt
   * pos.txt
   * sentiment.marshal
   * sentiment.marshal.3
* sim
* summary
* tag
* untils
* \__init\__.py

## 源码分析
1、在snownlp中，查看sentiments方法，发现，sentiments中，调用了classify;</br>
代码如下：
```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from . import normal
from . import seg
from . import tag
from . import sentiment
from .sim import bm25
from .summary import textrank
from .summary import words_merge


class SnowNLP(object):

    def __init__(self, doc):
        self.doc = doc
        self.bm25 = bm25.BM25(doc)

    @property
    def words(self):
        return seg.seg(self.doc)

    @property
    def sentences(self):
        return normal.get_sentences(self.doc)

    @property
    def han(self):
        return normal.zh2hans(self.doc)

    @property
    def pinyin(self):
        return normal.get_pinyin(self.doc)

    @property
    def sentiments(self):
        return sentiment.classify(self.doc)  #看到，sentiment调用classify方法；
        '''
        '''
  ```
  2、sentiment文件夹
  sentiment中，先创建了Sentiment对象  
  调用load方法，加载已经训练好的数据字典，再调用classify方法，在classify中，实际调用Bayes对象中的classify方法；  
  
  代码如下：
  ```python
  # -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import codecs

from .. import normal
from .. import seg
from ..classification.bayes import Bayes

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'sentiment.marshal')


class Sentiment(object):

    def __init__(self):
        #创建Bayes对象
        self.classifier = Bayes()

    #保存训练好的字典数据    
    def save(self, fname, iszip=True):
        self.classifier.save(fname, iszip)

    #加载字典数据
    def load(self, fname=data_path, iszip=True):
        self.classifier.load(fname, iszip)

    #对文档进行分词
    def handle(self, doc):
        words = seg.seg(doc)
        words = normal.filter_stop(words)
        return words

    #训练数据集
    def train(self, neg_docs, pos_docs):
        data = []
        #读取消极评论list，同时为每条评论加上neg标签，也放入一个list中
        for sent in neg_docs:
            data.append([self.handle(sent), 'neg'])
        #读取积极评论list，同时为每条评论加上pos标签，也放入一个list中
        for sent in pos_docs:
            data.append([self.handle(sent), 'pos'])
        #调用分类器的训练数据集的方法，对模型进行训练
        self.classifier.train(data)
        #此处的train方法，调用了classifier的train
        
    #分类
    def classify(self, sent):
        #调用贝叶斯分类器的分类方法，获取分类标签和概率
        ret, prob = self.classifier.classify(self.handle(sent))
        #如果分类的标签是pos，直接返回概率值
        if ret == 'pos':
            return prob
        #否则为neg标签，返回1-prob
        return 1-prob


classifier = Sentiment()
classifier.load()


#训练数据
def train(neg_file, pos_file):
    #打开消极数据文件和积极数据文件，读入neg_docs和pos_docs
    neg_docs = codecs.open(neg_file, 'r', 'utf-8').readlines()
    pos_docs = codecs.open(pos_file, 'r', 'utf-8').readlines()
    #训练数据，传入积极、消极评论的list
    classifier.train(neg_docs, pos_docs)


#保存数据字典
def save(fname, iszip=True):
    classifier.save(fname, iszip)


#加载数据字典
def load(fname, iszip=True):
    classifier.load(fname, iszip)


#对语句进行分类
def classify(sent):
    return classifier.classify(sent)
  ```

在sentiment中，包含了训练数据集的方法，sentiment文件夹中的neg.txt和pos.txt是已经分类好的评论数据，neg.txt是消极评论，pos.txt是积极评论；  
sentiment.marshal和sentiment.marshal.3中存放的是序列化后的数据字典  
在train()方法中，首先读取消极和积极评论txt文件，然后获取每一条评论，放入到list集合中  
训练的代码：
```python
#训练数据
def train(neg_file, pos_file):
    #打开消极数据文件和积极数据文件，读入neg_docs和pos_docs
    neg_docs = codecs.open(neg_file, 'r', 'utf-8').readlines()
    pos_docs = codecs.open(pos_file, 'r', 'utf-8').readlines()
    #训练数据，传入积极、消极评论的list
    classifier.train(neg_docs, pos_docs)
```
在这里，有调用了Sentiment对象的train()方法；  
在train方法中，遍历了传入的积极、消极评论list，为每条评论进行分词，并为加上了分类标签，此时的数据格式如下：</br>
评论分词后的数据格式：['收到','没有'...]</br>
加上标签后的数据格式(以消极评论为例)：[ [['收到','没有' ...],'neg'] ,  [['小熊','宝宝' ...],‘neg’] ........]]</br>
可以看到每一条评论都是一个list，其中又包含了评论分词后的list和评论的分类标签</br>
```python
    #训练数据集
    def train(self, neg_docs, pos_docs):
        data = []
        #读取消极评论list，同时为每条评论加上neg标签，也放入一个list中
        for sent in neg_docs:
            data.append([self.handle(sent), 'neg'])
        #读取积极评论list，同时为每条评论加上pos标签，也放入一个list中
        for sent in pos_docs:
            data.append([self.handle(sent), 'pos'])
        #调用分类器的训练数据集的方法，对模型进行训练
        self.classifier.train(data)
        #此处的train方法，调用了classifier的train
```  
3、classification文件中的bayes.py  
代码如下：
```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys
import gzip
#为了下次计算概率时，不用重新训练，可以将训练得到的数据序列化到文件中，下次直接加载文件，将文件反序列为对象，从对象中获取数据即可(save和load方法)
#得到了训练的数据后，可以使用朴素贝叶斯分类对进行分类了
import marshal  #存放的序列化的字典数据
from math import log, exp

from ..utils.frequency import AddOneProb


class Bayes(object):

    #在bayes对象中，有两个属性d和total,d是一个数据字典，total存储所有分类的总词数，
    #经过train方法训练数据集后，d中存储的是每个分类标签的数据key为分类标签，value是一个AddOneProb对象。
    def __init__(self):
        #标签数据对象
        self.d = {}
        #所有分类的词数之和
        self.total = 0

    #保存字典数据
    def save(self, fname, iszip=True):
        #创建对象，用来存储训练结果
        d = {}
        #添加total，也就是积极消极评论分词的总词数
        d['total'] = self.total
        #d为分类标签，存储的每个标签的数据对象
        d['d'] = {}
        for k, v in self.d.items():
            #k为分类标签，v为标签对应的所有分词数据，是一个AddOneProb对象
            d['d'][k] = v.__dict__
        #这里判断python的版本
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        #这里可以有两种方法可以选择进行存储
        if not iszip:
            #将序列化后的二进制数据，直接写入文件
            marshal.dump(d, open(fname, 'wb'))
        else:
            f = gzip.open(fname, 'wb')
            f.write(marshal.dumps(d))
            f.close()

    #加载数据字典
    def load(self, fname, iszip=True):
        #判断版本
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        #判断打开文件的方式
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = marshal.loads(f.read())
            f.close()
        #从文件中读取数据，为total和d对象赋值
        self.total = d['total']
        self.d = {}
        for k, v in d['d'].items():
            self.d[k] = AddOneProb()
            self.d[k].__dict__ = v

    #训练数据集
    def train(self, data):
        #遍历数据集
        for d in data:
            #d[1]标签--->分类类别
            c = d[1]
            #判断数据字典中是否有当前的标签
            if c not in self.d:
                #如果没有该标签，加入标签，值是一个AddOneProb对象
                self.d[c] = AddOneProb()
            #d[0]是评论分词list，遍历分词list
            for word in d[0]:
                #调用AddOneProb中的add方法，添加单词
                self.d[c].add(word, 1)
        #计算总词数
        self.total = sum(map(lambda x: self.d[x].getsum(), self.d.keys()))

    #贝叶斯分类
    def classify(self, x):
        tmp = {}
        #遍历每个分类标签
        for k in self.d:
            #获取每个分类标签下的总词数和所有标签总词数，求对数差相当于log（某标签下的总词数/所有标签总词数）
            tmp[k] = log(self.d[k].getsum()) - log(self.total)
            for word in x:
                #获取每个单词出现的频率，log[（某标签下的总词数/所有标签总词数）*单词出现频率]
                tmp[k] += log(self.d[k].freq(word))
        #计算概率，由于直接得到的概率值比较小，这里应该使用了一种方法来转换，原理还不是很明白
        ret, prob = 0, 0
        for k in self.d:
            now = 0
            for otherk in self.d:
                now += exp(tmp[otherk]-tmp[k])
            now = 1/now
            if now > prob:
                ret, prob = k, now
        return (ret, prob)
```
在bayes对象中，有两个属性d和total,d是一个数据字典，total存储所有分类的总词数，经过train方法训练数据集后，d中存储的是每个分类标签的数据key为分类标签，value是一个AddOneProb对象。在AddOneProb对象中，同样存在d和total属性，这里的total存储的是每个分类各自的单词总数，d中存储的是所有出现过的单词，单词作为key，单词出现的次数作为value.为了下次计算概率时，不用重新训练，可以将训练得到的数据序列化到文件中，下次直接加载文件，将文件反序列为对象，从对象中获取数据即可(save和load方法)  
4、得到了训练的数据后，使用朴素贝叶斯分类对进行分类
