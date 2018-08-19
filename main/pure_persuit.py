#!/usr/bin/python
# -*- coding: UTF8 -*-

""" 追踪点轨迹测试，纯最终算法
    https://blog.csdn.net/AdamShan/article/details/80555174
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

k = 0.25  # 前视距离与车速的系数
Lfc = 2.0  # 最小前视距离
L = 2.34  # 车辆轴距，单位：m

# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True, "Car1")
car_controls = airsim.CarControls()
client.reset()

class VehicleState:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):
    """1.更新车辆的位置和车头方向和车的速度，由于车是沿圆弧运动的，则车的速度方向与圆弧相切。
       因此每次更新时车沿圆心转过的角度为角速度乘上dt，又v/L*tan(delta)=v/L * L/R = v/R = w，
       因此第三个公式用的就是角速度乘上时间得到的车离轨迹圆心转过的角度。
       2.在弧形轨迹上，车头自身转过的角度yaw等于这条弧线对应的圆周角。
    """

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state

def PControl(target, current):
    """根据当前速度与目标速度的差值，设置一个加速度
    """
    a = Kp * (target - current)

    return a


def pure_pursuit_control(state, cx, cy, pind):
    """根据前视距离
    """
    # 找到目标点位置，目标点不是每次都会变的，取决于车离目标点的位置
    ind = calc_target_index(state, cx, cy)

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
    Lf = k * state.v + Lfc

    # atan2(y,x)返回坐标点的弧度，范围是-pi~pi,相当于就是arctan(y/x)
    # 这里第二个参数是x=1，返回的角度始终是-pi/2~pi/2
    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)

    return delta, ind

def calc_target_index(state, cx, cy):
    # 搜索最临近的路点，搜索策略太复杂，需要修改
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)] # 计算到每个点的距离
    ind = d.index(min(d)) # 获取最邻近点在数组中的位置
    L = 0.0

    Lf = k * state.v + Lfc # 根据速度计算出前视距离

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

def speedControl(speed):
    """控制车速，速度快了采刹车，速度慢了采油门
    """
    car_state = client.getCarState()
    
    if (car_state.speed < speed):
        car_controls.throttle = 1
    else:
        car_controls.throttle = 0

    client.setCarControls(car_controls)

def getSimpleCarState():
    car_state = client.getCarState()
    x_val, y_val, _ = getCarPosition(car_state)
    yaw = getCarOrientation(car_state)
    return VehicleState(x = x_val, y = y_val, yaw = yaw, v=car_state.speed)

def executeCarControls(delta):
    k = 3
    car_controls.steering = k * delta
    client.setCarControls(car_controls)

def main():
    car_speed = 35 / 3.6  # 40km/s

    # 追踪坐标
    cx = np.arange(0, 200, 0.5)
    cy = [math.sin(ix / 10.0) * ix / 2.0 for ix in cx]

    # track = common.read_list_from_file("main/Track.txt")

    # cx = [i[0]/100.0 for i in track]
    # cy = [i[1]/100.0 for i in track]

    state = getSimpleCarState()

    x = [state.x]
    y = [state.y]

    target_ind = calc_target_index(state, cx, cy)

    while True:
        # 1.控制车速
        speedControl(car_speed)

        # 
        state = getSimpleCarState()
        x.append(state.x)
        y.append(state.y)
        
        di, target_ind = pure_pursuit_control(state, cx, cy, target_ind)
        executeCarControls(di)
        print (target_ind)

        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.plot(cx[target_ind], cy[target_ind], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        plt.pause(0.001)


if __name__ == '__main__':
    main()
