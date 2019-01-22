#!/usr/bin/env python 
#!-*-coding:utf-8 -*-
#!@Time		: 2019/1/22 0022	21:25
#!@Author	: XW_Jia
#!@File		: set.py
'''
练习1： 关系运算
有如下两个集合，pythons是报名python课程的学员名字集合，ais是报名人工智能课程的学员名字集合
pythons={'albert','孙悟空','周星驰','朱茵','林志玲'}
ais={'猪八戒','郭德纲','林忆莲'，'周星驰'}
1. 求出即报名python又报名ai课程的学员名字集合
2. 求出所有报名的学生名字集合
3. 求出只报名python课程的学员名字
4. 求出没有同时这两门课程的学员名字集合
'''
# 求出即报名python又报名ai课程的学员名字集合
print(pythons & ais)
# 求出所有报名的学生名字集合
print(pythons | ais)
# 求出只报名python课程的学员名字
print(pythons - ais)
# 求出没有同时这两门课程的学员名字集合
print(pythons ^ ais)