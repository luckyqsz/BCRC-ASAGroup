Albert_age=18

count=0
while count < 3:
    guess=int(input('>>: '))
    if guess == Albert_age:
        print('you got it')
        break
    count+=1
print('---------------')

'''
#8：猜年龄游戏升级版 
要求：
    允许用户最多尝试3次
    每尝试3次后，如果还没猜对，就问用户是否还想继续玩，如果回答Y或y, 就继续让其猜3次，以此往复，
    如果回答N或n，就退出程序
    如何猜对了，就直接退出
'''
Albert_age = 18
count = 0
while True:
    if count == 3:
        choice = input('继续（Y/N）? >>:')
        if choice == 'Y' or choice == 'y':
            count = 0
        else:
            break

    guess = int(input('猜一个年龄：'))
    if guess == Albert_age:
        print('正确，结束')
        break
    count += 1
