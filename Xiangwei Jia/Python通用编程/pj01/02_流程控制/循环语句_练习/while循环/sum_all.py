#5. 求1-2+3-4+5 ... 99的所有数的和
count = 1
res = 0
while count <= 99:
    if count % 2 == 0:
        res -= count
    else:
        res += count
    count += 1
print(res)