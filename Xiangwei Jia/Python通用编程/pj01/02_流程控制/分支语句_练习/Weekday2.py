today=input('>>: ')
if today == 'Saturday' or today == 'Sunday':
    print('出去浪')

elif today == 'Monday' or today == 'Tuesday' or today == 'Wednesday' or today == 'Thursday' or today == 'Friday':
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