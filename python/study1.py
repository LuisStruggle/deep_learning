#!/usr/bin/env python3
# -*- coding: utf-8 -*-

a = range(100)
print(list(a[::3]))

print('a' < 'b' < 'c')

for i in range(2):
    print(i)
for i in range(4, 6):
    print(i)

print(3 > 2 > 2)

nums = set([1, 1, 2, 2, 3, 3])

print(nums)
"""
while True:
    pass
"""

numbers = [1, 2, 3, 4]
numbers.append([5, 6, 7, 8])
print(numbers)
print(len(numbers))
"""
x = 1


def change(a):
    x += 1
    print(x)


change(x)
"""


class Person:
    def __init__(self):
        pass

    def getAge(self):
        print(__name__)


p = Person()
p.getAge()

print('GNU\'s Not %s %%' % 'UNIX')
