my_age = 24
guess_age = int(input("Please guess a number:"))

while guess_age != my_age:

    if guess_age > my_age:
        print("Guess too large!")
    else:
        print("Guess too small!")
    guess_age = int(input("Please guess a number:"))

print("Congratulation~")

