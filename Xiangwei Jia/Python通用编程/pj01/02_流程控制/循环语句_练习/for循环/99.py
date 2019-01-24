#打印99乘法表

#range是左闭右开
for i in range(1,10):
    for j in range(1, i+1):
        print('%s*%s=%s' %(i, j, i*j),end=' ')
    print()