import numpy as np
import math
import matplotlib.pyplot as plt

k = 0.1  # 前视距离与车速的系数
Lfc = 2.0  # 最小前视距离
Kp = 1.0  # 速度P控制器系数
dt = 0.03  # 时间间隔，单位：s
L = 2.9  # 车辆轴距，单位：m

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

def main():
    #  设置目标路点
    cx = np.arange(0, 100, 1)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

    target_speed = 100 / 3.6  # [m/s]

    T = 100.0  # 最大模拟时间

    # 设置车辆的初始状态
    state = VehicleState(x=-5.0, y=5.0, yaw=0.0, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy)

    while T >= time and lastIndex > target_ind:
        ai = PControl(target_speed, state.v)
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
        plt.plot(x, y, "-b", label="trajectory")
        plt.plot(cx[target_ind], cy[target_ind], "go", label="target")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        plt.pause(0.001)
        # plt.show()

if __name__ == '__main__':
    main()