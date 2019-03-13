'''
文件“价格.txt”内容：每一行内容分别为商品名字，价钱，个数，求出本次购物花费的总钱数
apple 10 3
tesla 100000 1
mac 3000 2
lenovo 30000 3
chicken 10 3
'''
with open('F:/practice/python通用编程课程/pj01/05_文件处理/价格.txt') as read_f:
    list1 = read_f.readlines()
    print(list1)
    cost = 0
    for str in list1:
        str_list = str.split()
        print(str_list)
        cost += int(str_list[1]) * int(str_list[2])
print(cost)



