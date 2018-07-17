#coding: utf-8
import matrix
import pre
import numpy as np
import time

np.set_printoptions(threshold=np.inf)


def pretreatment(filepath_stop, path, pre_save_path, filename):
    '''加载停用词'''
    content = open(filepath_stop, 'r').read()
    stopwords = content.splitlines()

    ''' 文件操作，预处理操作 '''
    # 处理文本
    pretreat = pre.Pretreatment(path, stopwords)
    pretreat.readfile()
    # 保存
    pretreat.stringlistsave(pre_save_path, filename)


def train(path, rate):
    train = matrix.Matrix()
    train.divide(path, rate)
    train.create_matrix()
    train.train()


if __name__ == '__main__':
    print('begin')
    t1 = time.time()
    path = "/Users/liangsong/Desktop/Corpus/0.txt"                             # 原文本所处的文件夹位置 @
    print('语料库位置: %s\n' % path)

    print('预处理')
    filepath_stop = 'ChineseStopWords.txt'                                      # 停用词位置 @
    pre_save_path = '/Users/liangsong/Desktop' + '/' + '预处理try'               # 预处理后，存放文本的位置 @
    filename = '预处理.txt'
    # pretreatment(filepath_stop, path, pre_save_path, filename)
    print('完成预处理操作\n')

    print('分类')
    rate = 5                                                                      # 折叠数  @
    train(pre_save_path + '/' + filename, rate)
    t2 = time.time()
    print('花费时间：', t2-t1)
    print('end')
