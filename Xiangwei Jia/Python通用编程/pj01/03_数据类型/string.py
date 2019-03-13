#!/usr/bin/env python 
#!-*-coding:utf-8 -*-
#!@Time		: 2019/1/21 0021	21:10
#!@Author	: XW_Jia
#!@File		: string.py
hobbies = "music basketball"

print(hobbies[0:5:2])
print(hobbies[0:5:-2])
print(hobbies[11:5:-2])
print(hobbies[-1])
print(hobbies[-3])

print('A' not in 'Albert')

print('n, a me '.split(','))
print('n,/a /me'.split('/',1)) # 数字表示切割次数，默认全部切割
print('a|b|c'.rsplit('|',1))    #.rsplit()表示从右边开始切分

print('a'.join('1234'))
print('a'.join('1 '))
print('a'.join('1'))