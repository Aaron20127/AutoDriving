#!/usr/bin/python
# -*- coding: UTF8 -*-

import time
import threading
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys

import math
import gzip
import os.path
import random
import copy

def write_list_to_file(file, list):
    """将一个列表或字典转换成json字符号串存储到文件中"""
    obj_string = json.dumps(list)
    fo = open(file, "w")
    fo.write(obj_string)
    fo.close()
    return obj_string

def read_list_from_file(file):
    """将字符串转换成json对象，即列表或字典"""
    fo = open(file, "r")
    obj = json.loads(fo.read())
    fo.close()
    return obj

def plot_picture(matrix, cmap=None, title=None, axis=True):
    """绘制矩阵图片
        matrix 是列表，每个元素代表一个图片的像素矩阵
        title  是列表，每个元素代表标题
        cmap   是色彩
    """
    def get_subplot_region_edge(num):
        for i in range(10000):
            if num <= i*i: 
                return i

    total = len(matrix)
    edge = get_subplot_region_edge(total)
    plt.figure() 

    for i in range(total):
        ax = plt.subplot(edge, edge, i+1)  
        if title:
            ax.set_title(title[i], fontsize=14)

        if cmap:
            plt.imshow(matrix[i], cmap=cmap)
        else:
            plt.imshow(matrix[i])

        if not axis:
            plt.xticks([]) # 关闭图片刻度，必须放在imshow之后才生效
            plt.yticks([])

def plot_base(y_coordinate, x_coordinate = [], line_lable = [], 
            line_color = [], title = '', x_lable = '', y_lable = '',
            x_limit = [], y_limit = [], y_scale = 'linear', p_type = [],
            grad = False):
    """
    描述：画一幅坐标曲线图，可以同时有多条曲线
    参数：y_coordinate （y坐标值，二元列表，例如[[1,2,3],[4,5,6]]，表示有两条曲线，每条曲线的y坐标为[1,2,3]和[4,5,6]）
            x_coordinate  (x坐标值，同y坐标值，如果不提供x坐标值，则默认是从0开始增加的整数)
            line_lable   （每条曲线代表的意义，就是曲线的名称，没有定义则使用默认的）
            line_color    (曲线的颜色，一维列表，如果比曲线的条数少，则循环使用给定的颜色；不给定时，使用默认颜色；
                        更多色彩查看 http://www.114la.com/other/rgb.htm)
            title        （整个图片的名称）
            x_lable      （x轴的含义）
            y_lable       (y轴的含义)
            x_limit       (x坐标的显示范围)
            y_scale       (y轴的单位比例，'linear'常规，'log'对数)
            p_type        (类型：line线条，scatter散点)
            grad          (网格)
    """

    if (x_coordinate and (len(y_coordinate) != len(x_coordinate))):
        print ("error：x坐标和y坐标不匹配！")
        sys.exit()
    
    if (line_lable and  (len(y_coordinate) != len(line_lable))):
        print ("error：线条数和线条名称数不匹配，线条数%d，线条名称数%d！" % \
                (len(y_coordinate),len(line_lable)))     
        sys.exit()

    if not line_color:
        line_color = ['#9932CC', '#FF4040' , '#FFA933', '#CDCD00',
                        '#CD8500', '#C0FF3E', '#B8860B', '#AB82FF']
        # print "info: 未指定色彩，使用默认颜色！"

    if len(y_coordinate) > len(line_color):
        print ("warning: 指定颜色种类少于线条数，线条%d种，颜色%d种！" % \
                (len(y_coordinate),len(line_color)))

    # plt.figure(figsize=(70, 35)) 
    plt.figure() 
    ax = plt.subplot(111)

    # 如果没有给x的坐标，设置从0开始计数的整数坐标
    if not x_coordinate:
        x_coordinate = [range(len(y)) for y in y_coordinate]

    # 如果没有给线条名称，则使用默认线条名称
    if not line_lable:
        line_lable = ["line " + str(i) for i in range(len(y_coordinate))]

    # 如果没有指定图形类型，默认画线条line
    if not p_type:
        p_type = ["line" for y in y_coordinate]

    for i in range(len(y_coordinate)):
        if p_type[i] == 'line':
            ax.plot(x_coordinate[i], y_coordinate[i], color = line_color[i%len(line_color)], \
                    linewidth = 1.0, label = line_lable[i])      
        elif p_type[i] == 'scatter': 
            ax.scatter(x_coordinate[i], y_coordinate[i],  s = 90, c=line_color[i%len(line_color)],\
                        linewidth = 2.0, alpha=0.6, marker='+', label = line_lable[i])
        else:
            print ("error：Invalid p_type %s！" % (p_type[i]))
            sys.exit()

    ax.set_title(title) # 标题
    ax.set_xlabel(x_lable) # x坐标的意义
    ax.set_ylabel(y_lable) # y坐标的意义
    ax.set_yscale(y_scale) # 'linear','log'
    ### 自适应轴的范围效果更好
    if x_limit: ax.set_xlim(x_limit) # x坐标显示的范围
    if y_limit: ax.set_ylim(y_limit) # y坐标显示范围
    
    # plt.xticks()
    # plt.yticks()
    plt.legend(loc="best") # 线条的名称显示在右下角
    if grad: plt.grid(True) # 网格

    # plt.savefig("file.png", dpi = 200)  #保存图片，默认png     
    # plt.show()


def test():
    x = np.linspace(-1,1,10)
    y = np.sqrt(1-x**2)

    #1. 4个系数，最高次数3次，最高次在系数矩阵的第一个
    z = np.polyfit(x, y, 2)
    p3 = np.poly1d(z)

    #2. 30个系数，最高次数29次
    p30 = np.poly1d(np.polyfit(x, y, 30))

    xp = np.linspace(-2, 2, 100)
    _ = plt.plot(x, y, 'X', xp, p3(xp), '-', xp, p30(xp), '--')
    plt.ylim(-2,2)
    plt.show()

def isOrdered(list):
    """判断元素是否有序
    """
    return all(x<y for x, y in zip(list, list[1:])) or \
           all(x>y for x, y in zip(list, list[1:]))

def getVetorialAngle(x1, y1, x2, y2):
    """获取两个向量的夹角的弧度值
    """
    theta = abs(math.atan2(y1, x1) - math.atan2(y2, x2))
    if theta > math.pi:
        theta = 2*math.pi - theta
    return theta

def getCoordinatesVectorRotationAngle(x_list, y_list):
    """获取坐标旋转总的旋转角度
    """
    # 计算向量坐标
    x = [j - i for i, j in zip(x_list, x_list[1:])]
    y = [j - i for i, j in zip(y_list, y_list[1:])]

    num = (len(x)-1) # 计算的夹角总数
    total_angle = 0

    for i in range(num):
        total_angle += getVetorialAngle(x[i], y[i], x[i+1], y[i+1])

    return total_angle

def rotateCoordinates(theta, x, y):
    """将坐标研逆时针方向旋转theta弧度
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    R = np.mat([[cos,-sin],[sin, cos]]) #旋转矩阵
    co = np.mat([x,y]) # 坐标矩阵
    co = R*co

    x_n, y_n = co[0].tolist()[0], co[1].tolist()[0]
    return x_n, y_n


def convertToStandardFunctionPoints(x_list, y_list):
    """计算当前的x和y坐标是否能拟合成一个函数，不能则减小点，最后将起点和终点都旋转到同一y值
       1.如果x坐标有序，则执行第5步
       2.如果x坐标无序，y坐标有序，并执行第5步
       3.根据每连续两点形成的向量转过的角度和是否大于180度，判断这些点是否能经过坐标旋转
         拟合成函数。
       4.如果不能拟合成函数，则去掉数组最后的一个点，执行第3步。
       5.如果能拟合成函数，则以起始坐标点为原点坐标。将所有坐标旋转以起点和终点形成的向量
         和x轴正方向的夹角。最后返回形成的新的坐标。
       
       注意：1.如果所有的点的x值相等，则不是一个标准的函数，不能使用np.polyfit拟合，
              这时需要将所有的点旋转90度，使所有点y值相等，才是一个标准的函数。
            2.只有3个点则所有向量夹角之和不可能超过180
    """
    # 深拷贝
    x = copy.deepcopy(x_list)
    y = copy.deepcopy(y_list)

    # 至少需要3个坐标
    if len(x) < 2:
        print ("error：至少需要两个坐标才能转换！")
        sys.exit()

    ### 1.不能拟合，则减少坐标点
    while (len(x) > 3) and (not isOrdered(x)) and (not isOrdered(y)):
        if getCoordinatesVectorRotationAngle(x, y) > math.pi:
            x = np.delete(x,len(x)-1)
            y = np.delete(y,len(y)-1)
        else:
            break

    ### 2.将剩下的第一个坐标作为原点坐标，其他坐标绕该点旋转
    # 旋转的角度为终点到圆心向量和x轴正向的夹角
    x = np.array([i - x[0] for i in x]) # 移动到原点坐标
    y = np.array([i - y[0] for i in y]) 

    # 最后一个点到圆心的向量与x轴的夹角，向量在x轴上方则夹角是0~pi，
    # 向量在x轴下方则夹角是0~(-pi)
    theta = math.atan2(y[-1], x[-1])

    # 因为矩阵是逆时针旋转的，所以要将最后一个点旋转到
    x, y = rotateCoordinates(-theta, x, y)

    return x, y

def getCurvature():
    """曲率半径R = (1 + y1**2)**1.5 / y2
    """
    def first_derivative(x, theta_0, theta_1):
        """二次函数的一阶导数,theta_0是原二次项系数，theta_1是原一次项系数
        """
        return (2 * theta_0 * x + theta_1)

    def second_derivative(x, theta_0):
        """二次函数的二阶导数,theta_0是原二次项系数
        """
        base = np.ones([1,x.size])[0]
        return (base * 2 * theta_0)

    r = 10
    # total_deta = math.pi/4
    # deta = np.linspace(-total_deta, total_deta, 10)
    # x = r * np.cos(deta)
    # y = r * np.sin(deta)

    x = np.linspace(0,r,10)
    y = np.sqrt(r**2-x**2)

    x = np.linspace(0,r,9)
    y = list(np.zeros([1,9]))[0]

    theta = np.polyfit(x, y, 2)
    theta_0 = theta[0]
    theta_1 = theta[1]
    theta_2 = theta[2]

    x_average = (x[0]+x[-1])/2
    dx = (x[-1]-x[0])/3.4
    x_average = np.linspace(x_average-dx, x_average+dx, 10)

    y1 = first_derivative(x_average, theta_0, theta_1)
    y2 = second_derivative(x_average, theta_0)

    R = (1 + y1**2)**1.5 / np.absolute(y2)
    R_mean = np.mean(R)

    curvature = 1.0/R_mean

    p2 = np.poly1d(theta)
    xp = np.linspace(-r, r, 100)
    _ = plt.plot(x, y, 'X', xp, p2(xp), '-')
    plt.title("R = " + str(R_mean))
    plt.xlim(-r,r)
    plt.ylim(-r,r)
    plt.show()


getCurvature()
# test()

# x = [-1,-2,0,0]
# y = [1,2,2,1]

# r = 10

# y = np.linspace(0,r,9)
# x = list(np.zeros([1,9]))[0]

# x_n, y_n = convertToStandardFunctionPoints(x,y)

# _ = plt.plot(x, y, '-', x_n, y_n, '-')
# plt.show()