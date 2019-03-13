# 如果:今天是Monday,那么:上班
# 如果:今天是Tuesday,那么:上班
# 如果:今天是Wednesday,那么:上班
# 如果:今天是Thursday,那么:上班
# 如果:今天是Friday,那么:上班
# 如果:今天是Saturday,那么:出去浪
# 如果:今天是Sunday,那么:出去浪

today=input('>>: ')
if today == 'Monday':
    print('上班')
elif today == 'Tuesday':
    print('上班')
elif today == 'Wednesday':
    print('上班')
elif today == 'Thursday':
    print('上班')
elif today == 'Friday':
    print('上班')
elif today == 'Saturday':
    print('出去浪')
elif today == 'Sunday':
    print('出去浪')
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