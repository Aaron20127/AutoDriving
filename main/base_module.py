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
