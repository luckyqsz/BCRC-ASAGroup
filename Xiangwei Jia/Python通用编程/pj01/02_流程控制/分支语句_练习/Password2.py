'''
Albert --> 超级管理员
tom  --> 普通管理员
jack,rain --> 业务主管
其他 --> 普通用户
'''

name=input('请输入用户名字：')

if name == 'Albert':
    print('超级管理员')
elif name == 'tom':
    print('普通管理员')
elif name == 'jack' or name == 'rain':
    print('业务主管')
else:
    print('普通用户')