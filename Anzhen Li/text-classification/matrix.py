# coding:utf-8
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import re

'''chi2、TfIdf构造矩阵'''
'''朴素贝叶斯分类'''

np.set_printoptions(threshold=np.inf)

class Matrix(object):
    def __init__(self):
        self.y = []            # y
        self.X = np.arange(0)
        self.y_test = []            # y_test
        self.X_test = np.arange(0)
        self.N = 0

    '''构造矩阵，并划分测试集与训练集'''
    def divide(self, path, rate):
        contents, y = self.readfile(path)
        v = CountVectorizer()
        wordCount = v.fit_transform(contents)
        vect = wordCount.toarray()
        ### 划分 ###
        print('总文档数：', self.N)
        kf = StratifiedKFold(n_splits=rate, shuffle=True, random_state=1)
        for train_all_index, test_index in kf.split(vect, y):
            self.X, self.y = np.array(vect)[train_all_index], np.array(y)[train_all_index]
            self.X_test, self.y_test = np.array(vect)[test_index], np.array(y)[test_index]
            break

    '''读文件内容'''
    def readfile(self, path):
        content = []
        y = []
        with open(path, 'r') as file:
            text = file.read().split('820660C500')
            for text_list in text[1:]:
                text_list_filter = []
                text_list2 = text_list.strip().split('txttxttxttxt\t')
                i = 0
                num = text_list2[0].strip('\n')
                if (int(num) > 1) and (int(num) < 15):
                    for text_line in text_list2[1:]:
                        i += 1
                        self.N += 1
                        text = self.filter(text_line)
                        text_list_filter.append(text)
                        content.append(text)
                        y.append(num)
        return content, y

    def filter(self, text_line):
        wordlist = []
        words = re.findall('(\w*?)[ |\n|\]]', text_line, re.S)
        for word in words:
            if (word != '') and (word != '\t') and (word != '\n'):
                wordlist.append(word)
        text = ' '.join(wordlist)
        return text

    '''选取特征值，得到tfidf权重'''
    def create_matrix(self):
        ### CHI2,选前10% ###
        select = SelectPercentile(chi2, percentile=10)
        self.X = select.fit_transform(self.X, self.y)
        self.X_test = select.transform(self.X_test)
        ### tfidf ###
        tfidf = TfidfTransformer()
        self.X = tfidf.fit_transform(self.X, self.y).toarray()   # 最终获得的训练矩阵
        self.X_test = tfidf.transform(self.X_test)

    '''cross-validation, MultinomialNB'''
    def train(self):
        print('训练：')
        p_cross = 0
        r_cross = 0
        f1_cross = 0
        n_splits = 10                        # 十折交叉验证
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        i = 0
        classifier_text = MultinomialNB()
        for train_index, validation_index in kf.split(self.X, self.y):
            ## 划分训练集、验证集 ##
            i += 1
            X_train, y_train = np.array(self.X)[train_index], np.array(self.y)[train_index]
            X_validation, y_validation = np.array(self.X)[validation_index], np.array(self.y)[validation_index]
      
            classifier_text.fit(X_train, y_train)
            y_validate = classifier_text.predict(X_validation)
            p, r, f1, _ = precision_recall_fscore_support(y_validation, y_validate, average='macro')
            p_cross += p
            r_cross += r
            f1_cross += f1
        print("交叉验证精确率: {0}, 召回率: {1}, F1值: {2}\n".format(p_cross/n_splits, r_cross/n_splits, f1_cross/n_splits))

        print('测试：')
        test_validate = classifier_text.predict(self.X_test)
        p, r, f1, _ = precision_recall_fscore_support(self.y_test, test_validate, average='macro')
        print("测试精确率: {0}, 召回率: {1}, F1值: {2}\n".format(p, r, f1))






