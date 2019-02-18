#基于seek实现动态监测文件中是否有新内容的添加

import time
with open('F:/practice/python通用编程课程/pj01/05_文件处理/test.txt','rb') as f:
    f.seek(0,2)
    while True:
        line=f.readline()
        if line:
            print(line.decode('utf-8'))
        else:
            time.sleep(0.2)