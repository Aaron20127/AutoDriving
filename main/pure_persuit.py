#!/usr/bin/python
# -*- coding: UTF8 -*-

""" 追踪点轨迹测试，纯最终算法
    https://blog.csdn.net/AdamShan/article/details/80555174

    1.汽车最小转弯半径4.5米，曲率为0.22，
"""
import setup_path
import airsim
import common

import json
import time
import os

import numpy as np
import math
import matplotlib.pyplot as plt
import copy

k = 0.05  # 前视距离与车速的系数
Lfc = 2.0  # 最小前视距离
L = 2.34  # 车辆轴距，单位：m

Kp = 1.0 # 油门系数

# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True, "Car1")
car_controls = airsim.CarControls()
client.reset()

class VehicleState:

    def __init__(self, x, y, yaw, v):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def PControl(target, current):
    """根据当前速度与目标速度的差值，设置一个加速度
    """
    a = Kp * (target - current)

    return a


def pure_pursuit_control(state, cx, cy, pind, curvs):
    """根据前视距离
    """
    # 找到目标点位置，目标点不是每次都会变的，取决于车离目标点的位置
    ind = calc_target_index(state, cx, cy, curvs)

    # 如果
    if pind >= ind:
        ind = pind

    # 获取目标点坐标
    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1

    # 计算车头与前视距离的夹角,要得到正确方向向量，必须是目标点的坐标减去车自身的位置坐标
    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw

    # ???
    if state.v < 0:  # back
        alpha = math.pi - alpha

    # 注意，前视距离不是车道跟踪点的距离，而是自己定义的一个距离。前视距离的长短决定了
    # 预设的弧线的曲率半径
    curv = curvs[ind]

    # if curv > 0.01:
    #     Lf = 0.01 * state.v + 1
    # else:
    #     Lf = 0.05 * state.v + Lfc
    Lf = k * state.v + Lfc

    # atan2(y,x)返回坐标点的弧度，范围是-pi~pi,相当于就是arctan(y/x)
    # 这里第二个参数是x=1，返回的角度始终是-pi/2~pi/2
    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)

    return delta, ind

def calc_target_index(state, cx, cy, curvs):
    # 搜索最临近的路点，搜索策略太复杂，需要修改
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)] # 计算到每个点的距离
    ind = d.index(min(d)) # 获取最邻近点在数组中的位置
    L = 0.0

    # 根据曲率半径计算当前的前视距离，前视距离越大，跟踪的前方的点越远
    if curvs[ind] > 0.01:
      Lf = 0.01 * state.v + 1
        # Lf = 0.0
        # ind += 1
    else:
      Lf = 0.05 * state.v + Lfc

    # Lf = 0.05 * state.v + Lfc


    # 上边找到的最近的点可能不是在车的前方，所以要再向前找几个点保证下一个目标点在车辆前方
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind + 1] - cx[ind]
        dy = cy[ind + 1] - cy[ind]
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1

    return ind


def getCarPosition(car_state):
    x_val = car_state.kinematics_estimated.position.x_val
    y_val = car_state.kinematics_estimated.position.y_val
    z_val = car_state.kinematics_estimated.position.z_val
    return x_val, y_val, z_val

def getCarOrientation(car_state):
    """将弧度转换成0-360度，注意原来的弧度是-3.14到3.14，
       且y坐标的正向朝下，角度正方向为顺时针方向
       return: 0-360
    """
    _, _, arc = airsim.to_eularian_angles(car_state.kinematics_estimated.orientation)

    return arc
    # if arc < 0:
    #     arc += 2*math.pi
    # return arc/(2*math.pi) * 360

def getSimpleCarState():
    car_state = client.getCarState()
    x_val, y_val, _ = getCarPosition(car_state)
    yaw = getCarOrientation(car_state)
    return VehicleState(x = x_val, y = y_val, yaw = yaw, v=car_state.speed)

def executeCarControls(delta, target_speed, curv):
    # 1.方向盘
    k = 2
    car_controls.steering = k * delta

    # 2.控制车速，速度快了采刹车，速度慢了采油门
    global Kp
    # turning_speed = 20 / 3.6
    turning_speed = target_speed
    car_state = client.getCarState()
    
    # if curv > 0.01:
    #     target_speed = turning_speed
    #     if Kp > 0.5:
    #         Kp -= 0.01
    # else:
    #     if Kp < 1.0:
    #         Kp += 0.001

    if (car_state.speed < target_speed):
        car_controls.throttle = Kp
    else:
        car_controls.throttle = 0

    # 3.控制车速，速度快了采刹车
    d_speed = car_state.speed - target_speed
    if d_speed > 0:
        car_controls.brake = 0.15 * d_speed
    else:
        car_controls.brake = 0

    client.setCarControls(car_controls)

def generatingTrackCoordinates(curv):
    """生成各种类型轨迹坐标
    """
    # 追踪坐标
    # cx = np.arange(0, 200, 2)
    # cy = [math.sin((ix+100) / 10.0) * (ix+100) / 2.0  for ix in cx]

    # track = common.read_list_from_file("main/Track.txt")

    # cx = [i[0]/100.0 for i in track]
    # cy = [i[1]/100.0 for i in track]

    # 圆形坐标测试
    r = 1.0 / curv
    num_points = 120
    total_deta = math.pi
    deta = np.linspace(-total_deta, total_deta, num_points, endpoint=False)
    cx = r * np.cos(deta) + r 
    cy = r * np.sin(deta)

    return cx, cy

def isTrackingFailure(center_x, center_y, car_x, car_y, r, error):
    """测试车是否偏离轨道，error表示允许的误差范围
    """
    state = False
    dev = abs(math.sqrt((center_x - car_x)**2 + (center_y - car_y)**2) - r)
    if dev > error:
        state = True
    
    return state, dev

def test_circle(cx, cy, center_x, center_y, r, error, target_speed, loop=3):

    cx = list(cx)
    cy = list(cy)
    
    state = getSimpleCarState()
    curvs = common.getCurvatureArray(cx, cy)
    target_ind = calc_target_index(state, cx, cy, curvs)
    x = [state.x]
    y = [state.y]

    trac_fail = False

    while loop > 0 and (not trac_fail):
        # 1.获取方向盘转角
        delta, target_ind = pure_pursuit_control(state, cx, cy, target_ind, curvs)

        # 2.将后边的坐标放到坐标列表最前边，使车可以循环行驶
        if target_ind + 2 > len(curvs) - 1:
            loop -= 1
            cx = cx[target_ind-1:len(cx)] + cx[0:target_ind-1]
            cy = cy[target_ind-1:len(cy)] + cy[0:target_ind-1]
            target_ind = 0

        # 3.控制车
        state = getSimpleCarState()
        executeCarControls(delta, target_speed, curvs[target_ind+2])

        # 4.判断车是否偏离轨道，车的位置到圆心的距离等于圆弧的半径
        trac_fail, dev = isTrackingFailure(center_x, center_y, state.x, state.y, r, error)

        # 5.打印信息
        x.append(state.x)
        y.append(state.y)

        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.plot(cx[target_ind], cy[target_ind], "go", label="target")
        # plt.axis("equal")
        plt.grid(True)
        plt.title("speed:%0.2f, curv:%0.5f, ste:%0.3f, thr:%0.3f,bre:%0.3f, ind:%d, loop:%d, dev:%0.2f" %\
                 (state.v * 3.6, curvs[target_ind], 
                  car_controls.steering, car_controls.throttle,
                  car_controls.brake, target_ind, loop, dev))
        plt.pause(0.001)

    return (not trac_fail)

def main():
    # target_speed = 12 / 3.6  # 目标速度
    # curv = 0.222             # 曲率
    # r = 1.0 / curv           # 曲率半径
    # error = 0.3              # 车偏离中线的误差范围

    # cx, cy = generatingTrackCoordinates(curv)
    # state = test_main(cx, cy, r, 0, r, error, target_speed, loop=3)

    # 测试速度与曲率关系
    min_c = 0.004
    max_c = 0.222
    d_c = 0.001   # 每次曲率增量

    max_v = 100.0
    min_v = 12.0 
    d_v = 1.0    # 每次速度增量


    c_v = []

    front_v = max_v
    error = 0.3              # 车偏离中线的误差范围
    for curv in np.arange(min_c + d_c, max_c, d_c):
        r = 1.0 / curv 
        target_speed = front_v
        for i in range(int(front_v - min_v)):
            cx, cy = generatingTrackCoordinates(curv)
            client.reset()
            if test_circle(cx, cy, r, 0, r, error, target_speed / 3.6, loop=1):
                c_v.append([curv,target_speed])
                front_v = target_speed
                common.write_list_to_file("c_v.txt", c_v)
                break
            target_speed -= d_v

if __name__ == '__main__':
    main()
