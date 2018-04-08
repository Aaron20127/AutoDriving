#!/usr/bin/python
# -*- coding: UTF8 -*-

import numpy as np

msg = "你好"
print(msg);

haha = 1 + 2 + \
        3


print haha
print """jj\
kl"""

x = 'a'
y = 'b'
print x
print y

print x, y

print '------2-----'

print (complex(2,3))

print '------2------'
str = '0123456789'

print str[0]
print str[2:]   
print str[2:5]
print str * 2
print str + "TEST"

print '\n------------'

list = ['run', 786, 2.23, 'john', 70.2]
tinylist = [123, 'john']

print list[0]
print list[1:3]
print list[2:]
print tinylist * 2
print list + tinylist

print '\n------------'

tuple = ('ru', 1, 2.1, 'g', 4.3)
tinytuple = (123, 'fd')

print tuple
print tuple[0]
print tuple[1:3]
print tuple[2:]

print tinytuple * 2
print tuple + tinytuple

print '\n------------'

dict = {}
dict['one'] = "this is one"
dict[2] = "this is two"

tinydict = {'name': 'john', 'code': 6734, 'dept': 'sales'}

print dict['one']
print dict[2]
print tinydict
print tinydict.keys()
print tinydict.values()

print '\n------------'

a = 5.0
b = 3.0

print "a / b", a/b
print "a //b", a//b
print "a % b", a%b
print "a **b", a**b

print '\n------------'

if (2 <> 3):
    print 'a'
else:
    print 'b'

print '\n------------'

a = 2;
b = 1;

print (a and b)
print (a or b)
print not a

print '\n------------'

a = 10
b = 20
list = [1, 2, 3, 4, 5];

if (a in list):
    print "a in list"
else:
    print "a not in list"

if (b not in list):
    print "b not in list"
else: 
    print "b in list"

if (2 in list):
    print "2 in list"
else:
    print "2 not in list"

print '\n------------'

a = 20
b = 20

if (a is b):
    print "a is b"
elif (a not in b):
    print "b is not a"
else:
    print "a is not b"


print '\n------------'

a=0
b=1
if ( a > 0 ) and ( b / a > 2 ):
    print "yes"
else :
    print "no"

print '\n------------'

def my_print(args):
    print args

def move(n, a, b, c):
    my_print ((a, '-->', c)) \
    if n==1 \
    else (move(n-1,a,c,b) or move(1,a,b,c) or move(n-1,b,a,c))

move (3, 'a', 'b', 'c')

print '\n------------'

count = 0
while (count < 9):
    count = count + 1
    if (count == 2):
        continue
    # if (count == 8):
        # break
    print count,
else:
    print count

print '\n------------'

for i in 'Python':
    print i,

fruits = ['1', '2', '3']
for i in fruits:
    print i,

for i in range(len(fruits)): # 索引
    print fruits[i],

print '\n------------'

print "%s, %d" % ('wo', 2)

print u'Hello\u0020World !'

print '\n------------'

a = u'123'
if (a.isdecimal()): # Unicode变量有很多函数
    print 'ok'

print '\n------------'

list = [1, 2, 3]
list.append(4)
list.append(5)

print list
del list[4]
print list
print 'len:', len(list)
print 'list + list:', list + list
print 'list * 2:', list * 2
print 'list[-2]:', list[-2]

for i in list: 
    print i,

print '\n------------'

tup = (1)
print tup
del tup
# print tups

print '\n------------'
dict = {'name': 'Lee', 'name': 'mini'} #字典不允许犍出现两次
print dict['name']

print '\n------------'
import time
print time.time()
print time.localtime(time.time())
print time.asctime(time.localtime(time.time()))
print time.localtime()
print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print '\n------------'
def printme(str):
    "打印到输出设备" 
    print str
    return

printme("你好，printme成功！")

print '\n------------'
def ChangeInt(a):
    "改变整数值"
    a = 10
    return

b = 10
ChangeInt(b)
print b

def changme(list):
    "插入元组"
    list.append([1,2,3])
    print list
    return

mylist = [10]
changme(mylist)
print mylist

print '\n------------'
def medef(a, b = 10):
    "print"
    print a, b
    return

medef(1)
medef(1,2)

print '\n------------'
def printinfo(a, *var):
    "print"
    print a,
    for i in var:
        print i,
    return

print printinfo(1,2,3,4,5)

print '\n------------'
# Standard library
#import random

# Third-party libraries
import numpy as np

arr1 = np.random.randn(2,1)
print(arr1)
print('******************************************************************')
arr2 = np.random.rand(2,4)
print(arr2)


a = [np.random.randn(y, 1) for y in [1, 2, 1]] 
print a
print a[1][1][0]

print '\n------------'
sizes = [2,3,4];
a = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
print a

print zip([1,2],[3,4])

print '\n------------'
test_data = [1,2]
training_data = [2,3]
if test_data: n_test = len(test_data)
n = len(training_data)
for j in xrange(8):
    print j,

print '\n------------'
import mnist_loader

"""
training_data, validation_data, test_data = \
mnist_loader.load_data()

print len(training_data)
print len(training_data[0])
print len(training_data[0][0])
print len(training_data[0][1])

print '-----'
print len(training_data[1])
print training_data[1][0]

print '-----'
print len(validation_data)
print len(validation_data[0])
print len(validation_data[0][0])
print len(validation_data[0][1])

print '-----'
print len(test_data)
print len(test_data[0])
print len(test_data[0][0])
print len(test_data[0][1])

print training_data[0][0]
"""

"""
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

print '\n------------'
print len(training_data)
print len(training_data[0])
print len(training_data[0][0])
print training_data[0][1]

print '\n------------'
print len(validation_data)
print len(validation_data[0])
print len(validation_data[0][0])
print validation_data[0][1]

print '\n------------'
print len(test_data)
print len(test_data[0])
print len(test_data[0][0])
print test_data[0][1]

"""

print '\n------------'
for j in xrange(9):
    print j,

print '\n------------'
import random
list = [20, 16, 10, 5];
random.shuffle(list)
print list
random.shuffle(list)
print list

print '\n------------'
list = [0, 1, 2, 3, 4, 5, 6, 7]
print list

mini_batch_size = 3
n = len(list)
mini_batches = [list[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

print mini_batches

print '\n------------'
# a = [[1, 2, 3, 4, 5]]
# nabla_b = [np.zeros(b.shape) for b in a]

# print nabla_b

print '\n------------'
a1 = np.array([1,1,1]) 
a2 = np.array([2,3,4])
a3 = np.array([2,3,4])

print a1 + a2
print a1 * 2
print a2 ** 2

print a2[1]

a3 = np.array([[1, 2, 3], [4, 5, 6]])
print a3
print a3[0]
print a3[0,0]

a2 = np.array([2,3,4])
a3 = np.array([2,3,4])
print a2 * a3

print '\n------------'
a1 = np.zeros((2,2)) 
print a1, a1.shape

print '\n-----------'
a = np.array([[1,2], [3,4]])
nabla_b = [np.zeros(b.shape) for b in a]
print nabla_b

print '\n-----------'
a = [np.array([2]),np.array([2])]
print a
b = [np.array([1]),np.array([1])]
print b

print a + b

c = [na + nb for na, nb in zip(a, b)]
print c

print '\n-----------'
list = [0, 1, 2, 3, 4]
print np.argmax(list)

print '\n-----------'
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

print sigmoid(np.array([[1, 2, 3],[1, 2, 3]]))

print '\n-----------'
tr_d = np.array([[1, 2, 3]])
training_inputs = [np.reshape(x, (3, 1)) for x in tr_d]

print training_inputs

print '\n-----------'
tr_d = np.array([[1, 2, 3],[1,2,3]])
tr_2 = tr_d * 2
tr_3 = [1,2,3]
print tr_2
print tr_3 * 2

print '\n-----------'
tr_d = np.array([[1, 2, 3],[1,2,3],[1,2,3]])
td_2 = tr_d[:2]
td_3 = [td_2]
print td_2
print td_3

print '\n-----------'
a = np.array([[1,2,3],[1,2,3]])

print a*a
print np.dot(a,a.transpose())

print '\n-----------'
a = np.array([[1,2,3],[1,2,3]])
b = np.array([[1],[1]])

print a
print b
print a + b


print '\n-----------'
a = np.array([[[1],[1],[1]],[[2],[2],[2]]]);

print a
print a.shape

print [np.reshape(x, (1, 3)) for x in a]
print [x for x in a]

print '\n-----------'
a = np.array([1,2,3],dtype=int)
print a
print type(a)
print a.dtype.name
print a.size
print a.itemsize

print '\n-----------'
a = np.array([[1,2,3],[4,5,6]],dtype='int16')
print a.shape
print a
print type(a)
print a.dtype.name
print a.size
print a.itemsize
print a.ndim

print '\n-----------'
a = np.zeros((3,4))
print a.shape
print a
print type(a)
print a.dtype.name
print a.size
print a.itemsize
print a.ndim

print '\n-----------'
a = np.ones((2,3,4),dtype=np.int16)
print a

# 3维矩阵转换成二维矩阵
print '\n-----------'
b = np.array([[[1],[2],[3]],[[4],[5],[6]]])
print b
print b.reshape(2,3)
print b.reshape(2,3).transpose()

