#1. 使用while循环输出1 2 3 4 5 6     8 9 10
count = 1
while count <= 10:
    if count == 7:
        count += 1
        continue
    else:
        print(count)
    count += 1

print('---------------')

count = 1
while count <= 10:
    if count != 7:
        print(count)
    count += 1
