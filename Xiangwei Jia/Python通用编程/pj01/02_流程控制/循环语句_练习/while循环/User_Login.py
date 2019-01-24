count = 0
while count < 3:
    name = input('请输入用户名：')
    password = input('请输入密码：')
    if name == 'Albert' and password == '123':
        print('bingo')
        break
    else:
        print('用户名或密码错误，剩余','%d'%(2-count ),'次')
        count += 1