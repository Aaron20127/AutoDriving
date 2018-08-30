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
# import base_module as bml

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

def getVectorDistance(x, y):
    """得到向量的长度
    """
    return math.sqrt(x**2+y**2)

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

def getCurvature(x_list, y_list):
    """计算一系列坐标点的曲率半径和曲率
       1.曲率半径R = (1 + y1**2)**1.5 / y2
       2.曲率curvature = 1.0/R
    """
    # 1.坐标变换
    x, y = convertToStandardFunctionPoints(x_list, y_list)

    # 2.获取二次拟合系数，theta_0是二次项系数，theta_1是一次项系数，theta_0是二次项系数
    theta = np.polyfit(x, y, 2)
    theta_0 = theta[0]
    theta_1 = theta[1]
    # theta_2 = theta[2]

    # 3.计算x的中点
    K = 3.2 # 调节x范围的系数
    num = 10 # 选择的坐标数量
    x_average = (x[0]+x[-1])/2
    dx = (x[-1]-x[0])/K
    x_new = np.linspace(x_average-dx, x_average+dx, num)

    # 4.计算曲率半径和曲率
    y1 = 2 * theta_0 * x_new + theta_1 # 一阶导数
    y2 = 2 * theta_0  # 二阶导数

    R = (1 + y1**2)**1.5 / np.absolute(y2) # 计算曲率半径
    R_mean = np.mean(R) # 求曲率半径平均值

    # 返回曲率
    return 1.0/R_mean

def getCurvatureArray(x, y, loop=False, num=3):
    """
    计算一连串坐标轨迹点的曲率，返回所有点的曲率数组，实验后貌似3个点效果最好

    Parameters
        loop: 这个轨迹是否形成一个环形，如果是环形，则首尾的点连在一起计算曲率
        num:  每次计算曲率拟合的点的个数，必须是奇数，比如每次使用5个点，取中点第三个点的曲率。
                那么能计算曲率的点应该是第3到倒数第3个，前边和后边各有两个点不能计算曲率。
                由于曲线连续，所以将前第1,2个点的曲率设成第3个点的曲率，最后两个点的曲率
                设成倒数第三个点的曲率。
        x: x点坐标集合，一维数组，np.array类型
        y: y点坐标集合，一维数组，np.array类型
        
    Returns
        curvature: 曲率集合，一维数组，np.array类型
    """
    # 1.判断拟合点是否大于坐标总数, 判断拟合点数是否大于等于3，判断拟合点是否是奇数
    length = len(x)
    if length < num:
        print ("error：总的坐标点数小于每次拟合点数！")
        sys.exit()
    
    if num < 3:
        print ("error：每次的拟合点数必须大于等于3！")
        sys.exit()

    if num % 2 == 0:
        print ("error：拟每次拟合的坐标点数量必须是奇数！")
        sys.exit()

    # 2.根据是否是封闭的环，做不同的处理
    x_n = list(x) # 转换成list
    y_n = list(y)
    extra_length = int((num-1)/2) #要拟合完所有点，前后应该再增加几个点
    curvature = [] # 输出的曲率集合

    if loop:
        # 在前后加上相邻的点，使所有点都能拟合
        x_n = x_n[length - extra_length: ] + x_n + x_n[ :extra_length]
        y_n = y_n[length - extra_length: ] + y_n + y_n[ :extra_length]

        for i in range(length):
            curvature.append(getCurvature(x_n[i:(i+num)], y_n[i:(i+num)]))
    else:
        # 先计算能拟合的点，在将不能拟合的点的值设成离他最近的值
        for i in range(length-(num-1)):
            curvature.append(getCurvature(x_n[i:(i+num)], y_n[i:(i+num)]))

        head = [curvature[0] for i in range(extra_length)] # 前边应该加几个点
        rear = [curvature[-1] for i in range(extra_length)]
        curvature = head + curvature + rear # 全部点相加

    return np.array(curvature)

def getTargetSpeedFromCurvs(curvs, c_v_file='main/c_v.txt'):
    """根据坐标点的曲率列表，得到速度列表
    """
    # 1.获取曲率速度对照表
    c_v = read_list_from_file(c_v_file)
    c_list = np.array([i[0] for i in c_v])
    v_list = [i[1] for i in c_v]

    # 2.根据曲率列表，得到速度列表
    speed_list = []
    for curv in curvs:
        ind = np.searchsorted(c_list, curv)
        if ind == len(c_list):
            ind -= 1
        speed_list.append(v_list[ind])

    return speed_list

def testCurvature():
    def test(x,y,num,k):
        """每次拟合num个点，为了方便显示使用k将曲率调整大一点
        """
        curvature = getCurvatureArray(x, y, num=num)
        print(curvature)
        print(1.0/curvature)
        plt.figure()
        _ = plt.plot(range(len(x)), y, 'X', np.arange(0, len(curvature), 1), k*curvature, 'o')
        plt.title("num = %d " % (num))

    ## 圆形测试曲率
    # r = 10
    # total_deta = math.pi
    # deta = np.linspace(-total_deta, total_deta, 10, endpoint=False)
    # x = r * np.cos(deta)
    # y = r * np.sin(deta)

    ## 正弦函数测试曲率
    x = np.linspace(0,100,500)
    y = [math.sin(ix / 3.0) * ix / 2.0 for ix in x]

    test(x,y,3,20) # 每次拟合3个点的曲率
    test(x,y,5,20) # 每次拟合5个点
    test(x,y,7,20) # 每次拟合7个点

    plt.show()

if __name__ == '__main__':
    testCurvature()
