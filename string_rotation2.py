import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t_total = 1000 # Total time[s]
interval = 1
dt = 0.1 # 1ステップの時間[s]
t = np.arange(0, t_total, dt)

M = 10 #衛星の質量[kg]
m = 0.1 #単位長さの紐の質量[kg]
l = 1 # 単位長さの紐の長さ[m]
dl_dt = 0.1

n = 20
omega = np.pi / 180 * 5
radius = 5

# 初期条件
def initial_position(n, omega, radius):
    # 位置と速度と張力
    x0 = np.zeros(n * 3 + 2)
    for i in range (n):
        x0[i] = np.pi / 2
        x0[n + i] = omega
        x0[2 * n + i] = 0
    x0[n * 3 + 1] = np.pi / 2
    return x0

x0 = initial_position(n, omega, radius)

def calc_theta_dot2(mass, length, tension, num, n, theta_list):
    if num == 0:
        return 6 * tension / mass / length * (np.sin(theta_list[num + 1] - theta_list[num]))
    elif num == n - 1:
        return 6 * tension / mass / length * (np.sin(theta_list[num - 1] - theta_list[num]))
    else:
        return 6 * tension / mass / length * (np.sin(theta_list[num + 1] - theta_list[num]) + np.sin(theta_list[num - 1] - theta_list[num]))

def equation(x, t, n, omega, radius, dt, dl_dt, l, m, M):
    ret = np.zeros(n * 3 + 2)
    ret[0:n] = x[n:n*2]
    l_num = int(dl_dt * t / l)
    for i in range (n):
        if i > l_num:
            continue
        elif i == l_num:
            partial_length = dl_dt * t - l_num * l
            partial_mass = m * partial_length / l
            if partial_length * partial_mass == 0:
                continue
            ret[n + i] = calc_theta_dot2(partial_mass, partial_length, x[n * 3], i, n, x[0:n])
            ret[n * 2 + i] = dl_dt
            ret[n * 3] += M * partial_length * ret[n + i] * np.sin(x[i] - x[0])
        else:
            ret[n + i] = calc_theta_dot2(m, l, x[n * 3], i, n, x[0:n])
            ret[n * 3] += M * l * ret[n + i] * np.sin(x[i] - x[0])
    ret[n * 3] += M * radius * omega**2 * np.sin(omega * t  - np.pi / 2 - x[0])
    ret[n * 3] = (ret[n * 3] - x[n * 3]) / dt
    ret[n * 3 + 1] = omega
    return ret

x = odeint(equation,x0,t,args=(n, omega, radius, dt, dl_dt, l, m, M))

fig,ax = plt.subplots()
image, = ax.plot([],[], 'o-', lw=1)
ax.set_xlim(- 2 - radius - n, 2 + radius + n)
ax.set_ylim(-2 - radius - n, 2 + radius + n)
# ax.set_title('t={}'.format(t))
theta = np.arange(0,2*np.pi, 2*np.pi/360)
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
ax.plot(circle_x,circle_y, color='black')
ax.set_aspect('equal')


def update_anim(frame_num):
    x_axis = np.zeros(n + 1)
    y_axis = np.zeros(n + 1)
    for i in range (n + 1):
        if i == 0:
            x_axis[i] = radius * np.cos(x.T[n * 3 + 1][frame_num])
            y_axis[i] = radius * np.sin(x.T[n * 3 + 1][frame_num])
        else:
            x_axis[i] = x_axis[i - 1] + x.T[n * 3 - i][frame_num] * np.cos(x.T[n - i][frame_num])
            y_axis[i] = y_axis[i - 1] + x.T[n * 3 - i][frame_num] * np.sin(x.T[n - i][frame_num])
    image.set_data(x_axis,y_axis)
    return image,

anim = FuncAnimation(fig, update_anim,frames=np.arange(0, len(t)),interval=interval ,blit=True)
# anim.save("rotation.gif", writer='pillow')
plt.show()

