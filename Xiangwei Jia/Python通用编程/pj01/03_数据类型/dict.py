#!/usr/bin/env python 
#!-*-coding:utf-8 -*-
#!@Time		: 2019/1/22 0022	19:28
#!@Author	: XW_Jia
#!@File		: dict.py

d1 = {
    'name': 'albert',
    'age': 18,
}
d1.setdefault('name', 'Albert')
d1.setdefault('gender', 'male')
print(d1)


d1 = {
    'name': 'albert',
    'age': 18,
}
d1.update({'name': 'Albert', 'gender': 'male'}) # 注意传参方式的不同
print(d1)

d1 = {
    'name': 'albert',
    'age': 18,
    'gender': 'male','3':3,
}

a = d1.keys()
print(a)
print(list(a)[0])
a = d1.values()
print(a)
print(list(a)[0])
a = d1.items()
print(a)
print(list(a)[0])

msg_dic={
'apple':10,
'tesla':100000,
'mac':3000,
'lenovo':30000,
'chicken':10,
}
goods_l=[]
while True:
    for key,item in msg_dic.items():
        print('name:{name} price:{price}'.format(price=item,name=key))
    choice=input('商品>>: ').strip()
    if not choice or choice not in msg_dic:continue
    count=input('购买个数>>: ').strip()
    if not count.isdigit():continue
    goods_l.append((choice,msg_dic[choice],count))

    print(goods_l)


#有如下值集合 [11,22,33,44,55,66,77,88,99,90...]，将所有大于 66 的值保存至字典的第一个key中，将小于 66 的值保存至第二个key的值中
#即： {'k1': 大于66的所有值, 'k2': 小于66的所有值}
a={'k1':[],'k2':[]}
c=[11,22,33,44,55,66,77,88,99,90]
for i in c:
    if i>66:
        a['k1'].append(i)
    else:
        a['k2'].append(i)
print(a)


