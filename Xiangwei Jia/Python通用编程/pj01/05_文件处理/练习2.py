#修改文件内容，把文件中的mac都替换成linux
import os

with open('F:/practice/python通用编程课程/pj01/05_文件处理/价格.txt') as read_f, open('F:/practice/python通用编程课程/pj01/05_文件处理/价格_更改后.txt','w') as write_f:
    for line in read_f:
        #print(type(line))
        line = line.replace('mac','linux')
        write_f.write(line)

os.remove('F:/practice/python通用编程课程/pj01/05_文件处理/价格.txt')
os.rename('F:/practice/python通用编程课程/pj01/05_文件处理/价格_更改后.txt','F:/practice/python通用编程课程/pj01/05_文件处理/价格.txt')