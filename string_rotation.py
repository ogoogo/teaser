import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

interval = 10 #計算上の時間間隔[ms]
t = np.arange(0,20,interval/100)

k = 50 #バネ定数[N/m]
m = 0.1 #重りの質量[kg]
w = np.sqrt(k/m)
w0 = np.sqrt(k/m) #今回は共振を再現するためにw=w0
x0 = [0,-1*w**2*0.2] #初期条件(v,x). つまり重りを釣り合いの位置から20cm離れたところからスタート
f = 0 #外力の大きさ
a = 0 #減衰係数

n = 10
omega = np.pi / 180 * 5
radius = 1

# 初期条件
def initial_position(n, omega, radius):
    # 位置と速度
    x0 = np.zeros(n * 4)
    for i in range (n):
        x0[2 * i] = radius * np.sin(2 * i * np.pi / n)
        x0[2 * i + 1] = -radius * np.cos(2 * i * np.pi / n)
        x0[2 * n + 2 * i] = radius * omega * np.cos(2 * i * np.pi / n)
        x0[2 * n + 2 * i + 1] = radius * omega * np.sin(2 * i * np.pi / n)
    return x0

x0 = initial_position(n, omega, radius)

 
def equation(x, t, n, omega, radius):
    # 回転の方程式を実装
    ret = np.zeros(n * 4)
    ret[0:2*n] += x[2*n:4*n]
    for i in range (n):
        radius_i = np.sqrt(x[2 * i]**2 + x[2 * i + 1]**2)
        ret[2 * n + 2 * i] = radius_i * omega**2 * np.sin(2 * i * np.pi / n + omega * t)
        ret[2 * n + 2 * i + 1] = -radius_i * omega**2 * np.cos(2 * i * np.pi / n + omega * t)
    return ret

x = odeint(equation,x0,t,args=(n, omega, radius))
print(x0)

fig,ax = plt.subplots()
image, = ax.plot([],[], 'o-', lw=1)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
# ax.set_title('t={}'.format(t))

def update_anim(frame_num):
    x_axis = np.zeros(n)
    y_axis = np.zeros(n)
    for i in range (n):
        x_axis[i] = x.T[2 * i][frame_num]
        y_axis[i] = x.T[2 * i + 1][frame_num]
    image.set_data(x_axis,y_axis)
    return image,

anim = FuncAnimation(fig, update_anim,frames=np.arange(0, len(t)),interval=interval ,blit=True)
plt.show()
