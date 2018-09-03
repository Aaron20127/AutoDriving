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
import sys

k = 0.04  # 前视距离与车速的系数
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
      Lf = state.v / 260 * state.v + Lfc

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

def updateSimpleCarState():
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

    d_speed_thr = target_speed - car_state.speed 

    if (d_speed_thr > 0):
        car_controls.throttle = 0.3 + 1.0 * abs(d_speed_thr)
    else:
        car_controls.throttle = 0

    # 3.控制车速，速度快了采刹车
    d_speed_bre = car_state.speed - target_speed
    if d_speed_bre > 0:
        car_controls.brake = 0.5 * d_speed_bre
    else:
        car_controls.brake = 0

    client.setCarControls(car_controls)

def generatingCircleTrackCoordinates(curv, point_density = 0.5):
    """生成圆形坐标
       curv: 曲率
       point_density：点密度，表示多少米一个点
    """
    r = 1.0 / curv
    num = (2.0 * math.pi * r) // point_density # point_density表示多少米一个点
    total_deta = math.pi
    deta = np.linspace(-total_deta, total_deta, num, endpoint=False)
    cx = r * np.cos(deta) + r 
    cy = r * np.sin(deta)

    return cx, cy

def generatingOvalTrackCoordinates(min_c, max_c, point_density = 0.5):
    """生成椭圆坐标轨迹
       椭圆曲率求解：http://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFD2013&filename=CFXB201314002&uid=WEEvREdxOWJmbC9oM1NjYkZCbDZZNTBMZWYzOUhpaFRvbXJVOHpIZTBFSFY=$R1yZ0H6jyaa0en3RxVUd8df-oHi7XMMDo7mtKT6mSmEvTuk11l2gFA!!&v=MzA4NTVGWm9SOGVYMUx1eFlTN0RoMVQzcVRyV00xRnJDVVJMS2ZiK1JtRnlEblVyN0tKaXZUYkxHNEg5TE5xNDk=
       曲率最大位置：c(a,0) = a / b**2 = max_c
       曲率最小位置：c(0,b) = b / a**2 = min_c
       联立得长轴和短轴:
            a = max_c**(-1.0/3) * min_c**(-2.0/3)
            b = max_c**(-2.0/3) * min_c**(-1.0/3)

       椭圆周长公式：
            L1 = pi * (a + b)
            L2 = pi * (a**2 + b**2)**(0.5)
            L3 = 0.5*L1 + 0.5*L2 （最精确）

       max_c: 椭圆最小曲率，在短轴处
       max_c: 椭圆最大曲率，在长轴处
       point_density：点密度，表示多少米一个点
    """
    if min_c > max_c:
        print ("error：短轴曲率应该小于长轴曲率！")
        sys.exit()
    if max_c > 0.225:
        print ("error：超过最大曲率0.225！")
        sys.exit()
        
    # 1.计算长短轴
    a = max_c**(-1.0/3) * min_c**(-2.0/3) # 长轴
    b = max_c**(-2.0/3) * min_c**(-1.0/3) # 短轴

    # 2.计算椭圆周长
    pi = math.pi
    L1 = pi * (a + b)
    L2 = pi * (a**2 + b**2)**(0.5)
    L = 0.5*L1 + 0.5*L2

    # 3.生成坐标点
    num = L // point_density # point_density表示多少米一个点
    total_deta = math.pi
    deta = np.linspace(-total_deta, total_deta, num, endpoint=False)
    cx = a * np.cos(deta) + a
    cy = b * np.sin(deta) 
    # plt.plot(cx, cy, "o", label="course")
    # plt.title("a=%f, b=%f" % (a, b))
    # plt.show()
    return cx, cy, a, b

def getListNearOfLocalIndex(lst, index, left, right, auto_regulation = True):
    """找到列表的的index位置的前left和后right个元素，把它们按顺序链接在一起。
        这个过程把lst看成首位相连的圈
    参数：
        lst: 列表
        index: 从第几个位置开始找
        left: index前边多少个元素
        right: index后边多少个元素
        auto_regulation: 为True，如果lst的元素总数小于left + right + 1个，则返回lst
    返回：
        新的list
    """
    lst_len = len(lst)
    if (left + right + 1) > lst_len:
        if auto_regulation:
            return lst
        else:
            print ("error：获取的列表长度大于原列表长度！")
            sys.exit()

    lst_n = [lst[index]]
    # 1.获取左半部分
    if index >= left:
        lst_n = lst[index - left : index] + lst_n
    else:
        lst_n = (lst[index - left :] + lst[: index]) + lst_n      
    # 2.获取右半部分
    if (index + right) < lst_len:
        lst_n += lst[index+1 : index+1 + right]
    else:
        lst_n += (lst[index+1:] + lst[:right - (lst_len - (index+1))])
    return lst_n

def getTargetSpeedFromCurrentSpeed(speed_lst, index, cur_speed):
    """根据当前车速计算前视距离
    """
    left = 20 + int(cur_speed * 3.6)
    right = 20 + int(cur_speed * 3.6)
    lst_new = getListNearOfLocalIndex(speed_lst, index, left, right)
    # print (lst_new)
    return min(lst_new)

class trajectory:
    """显示生成的轨迹数据
    """
    @staticmethod
    def json_pack(cx, cy, car_x, car_y, t_x, t_y, title):
        j_data = {
            "cx" : cx,
            "cy" : cy,
            "car_x" : car_x,
            "car_y" : car_y,
            "t_x" : t_x,
            "t_y" : t_y,
            "title" : title
        }
        return j_data

    @staticmethod
    def json_parse(j_data):
        
        cx = j_data["cx"]
        cy = j_data["cy"]
        t_x = j_data["t_x"]
        t_y = j_data["t_y"]
        car_x = j_data["car_x"]
        car_y = j_data["car_y"]
        title = j_data["title"]
        return cx, cy, t_x, t_y, car_x, car_y, title
        
    @staticmethod
    def display(j_data):
        """显示整个道路轨迹和车的运动轨迹
        输入:
        j_data:json格式数据
        """
        cx, cy, t_x, t_y, car_x, car_y, title = trajectory.json_parse(j_data)

        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(car_x, car_y, "-b", label="trajectory")
        plt.plot(t_x, t_y, "go", label="target")
        # plt.axis("equal")
        plt.grid(True)
        plt.title(title)
        plt.pause(0.0001)

        # print(title)
        # print(car_x, car_y)

def testCircleRoad(cx, cy, center_x, center_y, r, error, target_speed, loop=3):
    '''测试圆环道路
    输入：
       cx,cy: 车的坐标
       center_x, center_y: 道路中心坐标
       r: 圆环道路的半径
       error: 允许车偏离道路的最大误差
       target_speed: 目标速度
       loop: 行驶的圈数
    返回：
       是否测试成功
    '''
    def isTrackingFailure(center_x, center_y, car_x, car_y, r, error):
        """测试车是否偏离轨道，error表示允许的误差范围
        """
        state = False
        dev = abs(math.sqrt((center_x - car_x)**2 + (center_y - car_y)**2) - r)
        if dev > error:
            state = True
        
        return state, dev

    cx = list(cx)
    cy = list(cy)
    
    state = updateSimpleCarState()
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
        state = updateSimpleCarState()
        executeCarControls(delta, target_speed, curvs[target_ind+2])

        # 4.判断车是否偏离轨道，车的位置到圆心的距离等于圆弧的半径
        trac_fail, dev = isTrackingFailure(center_x, center_y, state.x, state.y, r, error)

        # 5.打印信息
        x.append(state.x)
        y.append(state.y)

        # plt.cla()
        # plt.plot(cx, cy, ".r", label="course")
        # plt.plot(x, y, "-b", label="trajectory")
        # plt.plot(cx[target_ind], cy[target_ind], "go", label="target")
        # # plt.axis("equal")
        # plt.grid(True)
        # plt.title("speed:%0.2f, curv:%0.5f, ste:%0.3f, thr:%0.3f,bre:%0.3f, ind:%d, loop:%d, dev:%0.2f" %\
        #          (state.v * 3.6, curvs[target_ind], 
        #           car_controls.steering, car_controls.throttle,
        #           car_controls.brake, target_ind, loop, dev))
        # plt.pause(0.001)

        print("speed:%0.2f, curv:%0.5f, ste:%0.3f, thr:%0.3f, bre:%0.3f, ind:%d/%d, loop:%d, dev:%0.2f" %\
                 (state.v * 3.6, curvs[target_ind], 
                  car_controls.steering, car_controls.throttle,
                  car_controls.brake, target_ind, len(cx), loop, dev))


    return (not trac_fail)

def testOvalRoad(cx, cy, center_x, center_y, a, b, error, loop=1, file='files/c_v.txt'):
    '''测试圆环道路
    输入：
       cx,cy: 车的坐标
       center_x, center_y: 道路中心坐标
       a, b: 长轴和短轴
       error: 允许车偏离道路的最大误差
       target_speed: 目标速度
       loop: 行驶的圈数
    返回：
       是否测试成功
    '''
    def isTrackingFailure(x, y, center_x, center_y, error, a = a, b = b):
        """ 得到车当前离轨道的误差距离
        """
        # 1.车相对于椭圆圆心的坐标向量和初始向量
        x = x - center_x
        y = y - center_y

        # 2.得到向量夹角
        deta = math.atan2(y, x)

        # 3.得到误差
        ox = a * np.cos(deta) 
        oy = b * np.sin(deta)
        
        error_n = abs(common.getVectorDistance(x, y) -\
                  common.getVectorDistance(ox, oy))
        if error >= error_n:
            return False, error_n
        else:
            return True, error_n    

    cx = list(cx)
    cy = list(cy)
    
    state = updateSimpleCarState()
    curvs = common.getCurvatureArray(cx, cy)
    target_speed_list = \
        common.getTargetSpeedFromCurvs(curvs, c_v_file = file)

    target_ind = calc_target_index(state, cx, cy, curvs)
    x = [state.x]
    y = [state.y]

    trac_fail = False

    # 另开一个用于图形显示的进程
    process_display = common.NewPipProcess(trajectory.display)
    count = 0

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
        state = updateSimpleCarState()
        curv = curvs[target_ind+2]
        target_speed = getTargetSpeedFromCurrentSpeed(target_speed_list, target_ind+2, state.v) / 3.6
        # target_speed = (target_speed_list[target_ind+2]) / 3.6
        # target_speed = 12 / 3.6
        executeCarControls(delta, target_speed, curv)

        # 4.判断车是否偏离轨道，车的位置到圆心的距离等于圆弧的半径
        trac_fail, dev = isTrackingFailure(state.x, state.y, center_x, center_y, error)
        trac_fail = False

        # 5.打印信息
        x.append(state.x) # 行驶轨迹
        y.append(state.y)

        tar_x = [cx[target_ind]] # 追踪点
        tar_y = [cy[target_ind]]

        count += 1
        if count % 5 == 0:
            process_display.put(trajectory.json_pack(cx, cy, x, y, tar_x, tar_y,
                    ("sp:%0.2f, tsp:%0.2f, curv:%0.5f, ste:%0.3f, thr:%0.3f, bre:%0.3f, ind:%d/%d, loop:%d, dev:%0.2f, flk:%d" %\
                    (state.v * 3.6, target_speed * 3.6, curvs[target_ind], 
                    car_controls.steering, car_controls.throttle,
                    car_controls.brake, target_ind, len(cx), loop, dev, 20 + int(state.v * 3.6)))))

            # plt.cla()
            # plt.plot(cx, cy, ".r", label="course")
            # plt.plot(x, y, "-b", label="trajectory")
            # plt.plot(cx[target_ind], cy[target_ind], "go", label="target")
            # # plt.axis("equal")
            # plt.grid(True)
            # plt.title("sp:%0.2f, tsp:%0.2f, curv:%0.5f, ste:%0.3f, thr:%0.3f, bre:%0.3f, ind:%d/%d, loop:%d, dev:%0.2f, flk:%d" %\
            #          (state.v * 3.6, target_speed * 3.6, curvs[target_ind], 
            #           car_controls.steering, car_controls.throttle,
            #           car_controls.brake, target_ind, len(cx), loop, dev, 20 + int(state.v * 3.6)))
            # plt.pause(0.001)

        # print("sp:%0.2f, tsp:%0.2f, curv:%0.5f, ste:%0.3f, thr:%0.3f, bre:%0.3f, ind:%d/%d, loop:%d, dev:%0.2f" %\
        #          (state.v * 3.6, target_speed * 3.6, curvs[target_ind], 
        #           car_controls.steering, car_controls.throttle,
        #           car_controls.brake, target_ind, len(cx), loop, dev))

    return (not trac_fail)

def generateCarSpeedCurveParallelTable(file = 'files/c_v_new.txt'):
    '''测试车速度与曲率关系，产生对应的列表，并保存成文件。
       默认的速度范围是12.0-200.0km/s，精度0.1;
       曲率是0.0010-0.2250，精度0.0001。
    '''
    min_c = 0.2
    max_c = 0.2550
    d_c = 0.0001   # 每次曲率增量

    max_v = 200.0
    min_v = 12.0 
    d_v = 0.1     # 每次速度增量

    c_v = []      # 需要保存的速度曲率对照表
    front_c = max_c # 保存前一次最大曲率
    error = 0.35    # 车偏离中线的误差范围

    for v in np.arange(min_v, max_v + d_v, d_v):
        target_speed = v
        curv = front_c
        for c in np.arange(min_c, front_c + d_c, d_c):
            r = 1.0 / curv
            cx, cy = generatingCircleTrackCoordinates(curv)

            client.reset()
            time.sleep(0.2)
            client.reset()
            time.sleep(0.2)
            client.reset()
            time.sleep(0.2)

            if testCircleRoad(cx, cy, r, 0, r, error, target_speed / 3.6, loop=1):
                c_v.insert(0, [round(curv, 4), round(target_speed, 1)]) # 按曲率从小到大排序
                front_c = curv
                common.write_list_to_file(file, c_v)
                break
            curv -= d_c

def main():

    ## 1.测试圆形中的行驶
    # client.reset()
    # time.sleep(0.2)
    # client.reset()
    # time.sleep(0.2)

    # target_speed = 200 / 3.6  # 目标速度
    # curv = 0.0010           # 曲率
    # r = 1.0 / curv           # 曲率半径
    # error = 0.35              # 车偏离中线的误差范围

    # cx, cy = generatingTrackCoordinates(curv, point_density=0.5)
    # state = test_circle(cx, cy, r, 0, r, error, target_speed, loop=3)

    # 2.生成速度和曲率关系
    # generateCarSpeedCurveParallelTable()

    ## 3.测试椭圆中行驶
    min_c = 0.0001 # 椭圆的曲最小曲率
    max_c = 0.2250 # 椭圆的最大曲率
    error = 0.35   # 偏离轨道的误差

    cx, cy, a, b = generatingOvalTrackCoordinates(min_c, max_c)

    print(a, b)

    state = testOvalRoad(cx, cy, a, 0, a, b, error)


if __name__ == '__main__':
    """
    存在问题：1.油门的控制，如何拟人
             2.如何制定一个策略使能精确入弯和出弯
    """
    main()
