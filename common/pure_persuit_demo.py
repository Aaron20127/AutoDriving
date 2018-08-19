#!/usr/bin/python
# -*- coding: UTF8 -*-

""" 追踪点轨迹测试，纯追踪算法
    https://blog.csdn.net/AdamShan/article/details/80555174
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import base_module as bml

kc = 4    # 前视距离与曲率的系数
k = 0.05  # 前视距离与车速的系数
Lfc = 2 # 最小前视距离

Kp = 1.0  # 控制加速度为正的大小，使汽车加速
Kt = 1.0  # 控制加速度为负的大小，使汽车减速

dt = 0.02  # 时间间隔，单位：s
L = 2.9  # 车辆轴距，单位：m

turning_speed = 40 / 3.6 #最小转弯速度

class VehicleState:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, curv=[]):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.curv= curv


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

def PControl(target, current, curv):
    """计算加速度，判断应该加速还是减速。入弯的时候使加速度很大，刹车快，并减小加速度系数。
       这样，在出弯的时候加速度就很小，不易偏离轨道
    """
    global Kp
    k = Kp

    if curv > 0.01:
        target = turning_speed
        if target < current:
            k = 4
            if Kp > 0.05:
                Kp -= 0.05
        else:
            k = 0
    else:
        if Kp < 1.0:
            Kp += 0.02

    a = k * (target - current)

    
    return a


def pure_pursuit_control(state, cx, cy, pind):
    """根据前视距离计算出目标点和车前轮因该转过的角度
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
    # 预设的弧线的曲率半径，在弯道处减小Lf，可使车轮应转的角度更大，使转弯效果更好
    curv = state.curv[ind]

    if curv > 0.01:
        Lf = 0.01 * state.v + 0.5
    else:
        Lf = 0.05 * state.v + Lfc

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

    # 根据曲率半径计算当前的前视距离，前视距离越大，跟踪的前方的点越远
    if state.curv[ind+1] > 0.01:
        Lf = 0.01 * state.v + 0.5
    else:
        Lf = 0.05 * state.v + Lfc

    # Lf越大当前跟踪的点越靠前
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind + 1] - cx[ind]
        dy = cy[ind + 1] - cy[ind]
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1

    return ind

def main():
    # 设置目标轨迹
    cx = np.arange(0, 100, 0.3)
    cy = [math.sin(ix / 3.0) * ix / 2.0 for ix in cx]

    # 获取曲率
    curvature = bml.getCurvatureArray(cx, cy)


    target_speed = 200 / 3.6  # [m/s]

    T = 100.0  # 最大模拟时间

    # 设置车辆的初始状态
    state = VehicleState(x=59, y=23, yaw=1.0, v=target_speed, curv=curvature)

    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy)

    while T >= time and lastIndex > target_ind:
        ai = PControl(target_speed, state.v, state.curv[target_ind+4])
        di, target_ind = pure_pursuit_control(state, cx, cy, target_ind)
        state = update(state, ai, di)

        time = time + dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)


        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(cx, 5*curvature, ".b", label="curvature")
        plt.plot(x, y, "-b", label="trajectory")
        plt.plot(cx[target_ind], cy[target_ind], "go", label="target")


        # plt.axis("equal")
        plt.grid(True)
        plt.title("Speed[km/h]:%0.4f, curv:%0.8f" % (state.v * 3.6, curvature[target_ind]))
        plt.pause(0.001)
        # plt.show()

if __name__ == '__main__':
    main()