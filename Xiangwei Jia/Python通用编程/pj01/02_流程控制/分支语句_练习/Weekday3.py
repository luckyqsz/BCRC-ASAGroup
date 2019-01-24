#使用列表来操作

today=input('>>: ')
if today in ['Saturday','Sunday']:
    print('出去浪')
elif today in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
    print('上班')
else:
    print('''必须输入其中一种:
    Monday
    Tuesday
    Wednesday
    Thursday
    Friday
    Saturday
    Sunday
    ''')