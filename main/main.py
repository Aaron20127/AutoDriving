import setup_path 
import airsim

import json
import time
import os
import numpy as np
import math

# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True, "Car1")
car_controls = airsim.CarControls()
client.reset()

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

    if arc < 0:
        arc += 2*math.pi
    return arc/(2*math.pi) * 360

def getRelativeAngle(line_orientation, car_orientation):
    """得到车道线与车头之间的夹角,后续需要改进
    """
    return (car_orientation - line_orientation)

def test_1():
    set_speed = 11.11  # 40km/s
    ref_x = 2.1     # 延x轴行走的垂直参考线
    max_y_val = -2000 # 行驶到y的最大位置

    for i in np.linspace(1, 1, num=10):
        while True:
            # 1.控制车速
            car_state = client.getCarState()
            
            if (car_state.speed < set_speed):
                car_controls.throttle = 1
            else:
                car_controls.throttle = 0

            client.setCarControls(car_controls)

            # 2.根据位置控制方向盘
            car_state = client.getCarState()
            x_val, y_val, z_val = getCarPosition(car_state)
            car_orientation = getCarOrientation(car_state)

            relative_angle = getRelativeAngle(270, car_orientation)
            k_relative_angle = 90
            distance = abs(x_val-ref_x)
            k_distance = 30.0

            para_a = distance/k_distance
            para_b = relative_angle/k_relative_angle

            if x_val > ref_x: #车在中线右边
                if relative_angle > 0:
                    car_controls.steering = - para_a - para_b
                elif relative_angle < 0:
                    car_controls.steering = - para_a + para_b
            elif x_val < ref_x: #车在中线左边
                if relative_angle > 0:
                    car_controls.steering = para_a - para_b
                elif relative_angle < 0:
                    car_controls.steering = para_a + para_b

            client.setCarControls(car_controls)

            # 3.打印位置信息
            print("postion: %0.2f, %0.2f, throttle: %0.2f, steering: %0.2f, relative_angle: %0.2f, para_a: %0.2f, para_b: %0.2f"  %\
                 (x_val, y_val, i, car_controls.steering, relative_angle, para_a, para_b))

            # 4.退出当前循环，进入下一循环
            if y_val < max_y_val:
                client.reset()

test_1()

# Go forward + steer right
# car_controls.throttle = 0.5
# car_controls.steering = 1
# client.setCarControls(car_controls)
# time.sleep(0.5)   # let car drive a bit

# car_controls.throttle = 0
# car_controls.steering = 0
# client.setCarControls(car_controls)



# client.reset()

# client.enableApiControl(False)


            
