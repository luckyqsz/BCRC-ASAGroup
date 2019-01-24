albert_age = 18
while True:
    guess = int(input('guess albert\'s age:'))
    if guess > albert_age:
        print('Too big')
    elif guess < albert_age:
        print('Too Small')
    else:
        print('Bingo!')
        break