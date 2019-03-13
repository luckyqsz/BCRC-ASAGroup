'''
练习，要求如下：
    1 循环验证用户输入的用户名与密码
    2 认证通过后，运行用户重复执行命令
    3 当用户输入命令为quit时，则退出整个程序
'''
name = 'Albert'
password = '123'

while True:
    inp_name = input('Please enter your name:')
    inp_password = input('Please enter your password:')
    if inp_name == name and inp_password == password:
        print('输入正确！')
        while True:
            cmd = input('输入操作：')
            print('执行：%s' %cmd)
            if cmd != 'quit':
                continue
            else:
                break
                #使用tag来跳出整个外部大循环，而且这样写外部循环的break以及外部else的continue都可以省了
                #tag = False
                #continue
    else:
        print('用户名或者密码错误')
        continue
    break