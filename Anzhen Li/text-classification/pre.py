# -*-coding:utf-8-*-
import re
import thulac
from pyhanlp import *
import multiprocessing

'''对文件进行预处理操作'''

class Pretreatment(object):
    def __init__(self, path, stopwords):
        self.filepath = path
        self.text = []
        self.text_cut_all = {}
        self.stopwords = stopwords
        self.textlist = {}  # 记录每类下所有的文章


    '''读文件内容，同时进行预处理'''
    def readfile(self):
        # pool = multiprocessing.Pool()
        with open(self.filepath, 'r') as file:
            self.text = file.read().split('820660C500')
            for text_list in self.text[1:]:
                text_list_filter = []
                text_list2 = text_list.strip().split('txttxttxttxt\t')
                i = 0
                num = text_list2[0].strip('\n')
                for text_line in text_list2[1:]:
                    i += 1
                    text = self.filter(text_line)
                    text_list_filter.append(text)
                    # pool.apply_async(self.mult_filter, (text_line,))
                # pool.close()
                # pool.join()
                self.textlist[num] = text_list_filter
                print('预处理完成的是%s类，共%d项' % (num, i))


    '''一个文本的预处理（分词、过滤）'''
    def filter(self, text_line):
        text_cut_one = []
        text_cuts = HanLP.segment(text_line)                        # @@ HanLp切词
        for text_cut in text_cuts:
            text_cut = re.findall(r'(.*?)/', str(text_cut))[0]    # @@
            if text_cut not in self.stopwords:
                text_cut = re.sub(u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+',
                                  '', text_cut, re.S and re.M)
                if text_cut != '\t' and '\n':
                    text_cut_one.append(text_cut)
        text = ' '.join(text_cut_one)
        return text


    '''存文本'''
    def stringlistsave(self, save_path, filename):
        path = save_path + "/" + filename
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(path, "a+") as fp:
            for key in self.textlist:
                fp.write('820660C500' + key + '\n')
                for text in self.textlist[key]:
                    fp.write("txttxttxttxt\t%s " % text)
        print('预处理存储OK，存储位置：%s' % path)






