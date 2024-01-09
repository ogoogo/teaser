import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import jacobian
import linecache
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

l =  2*10**3  #m
m = 12/5 #kg
T = 207 #Nm

h = 1000 #s 時間刻み幅
time = 300000 #s　解析時間
N = int(time/h)
print(N)

def A(x1,x2,x3,x4,x5):
    Aarray = m * l**2/2 * np.array([[10             , 8*np.cos(x1-x2), 6*np.cos(x1-x3), 4*np.cos(x1-x4), 2*np.cos(x1-x5)],
                                    [8*np.cos(x2-x1), 8              , 6*np.cos(x2-x3), 4*np.cos(x2-x4), 2*np.cos(x2-x5)],
                                    [6*np.cos(x3-x1), 6*np.cos(x3-x2), 6              , 4*np.cos(x3-x4), 2*np.cos(x3-x5)],
                                    [4*np.cos(x4-x1), 4*np.cos(x4-x2), 4*np.cos(x4-x3), 4              , 2*np.cos(x4-x5)],
                                    [2*np.cos(x5-x1), 2*np.cos(x5-x2), 2*np.cos(x5-x3), 2*np.cos(x5-x4), 2              ]])
    return Aarray


def B(x1,x2,x3,x4,x5,dx1,dx2,dx3,dx4,dx5,T):
    l1 =                           - 8*dx1*dx2*np.sin(x1-x2) - 6*dx1*dx3*np.sin(x1-x3) - 4*dx1*dx4*np.sin(x1-x4) - 2*dx1*dx5*np.sin(x1-x5)
    l2 = - 8*dx2*dx1*np.sin(x2-x1)                           - 6*dx2*dx3*np.sin(x2-x3) - 4*dx2*dx4*np.sin(x2-x4) - 2*dx2*dx5*np.sin(x2-x5)
    l3 = - 6*dx3*dx1*np.sin(x3-x1) - 6*dx3*dx2*np.sin(x3-x2)                           - 4*dx3*dx4*np.sin(x3-x4) - 2*dx3*dx5*np.sin(x3-x5)
    l4 = - 4*dx4*dx1*np.sin(x4-x1) - 4*dx4*dx2*np.sin(x4-x2) - 4*dx4*dx3*np.sin(x4-x3)                           - 2*dx4*dx5*np.sin(x4-x5)
    l5 = - 2*dx5*dx1*np.sin(x5-x1) - 2*dx5*dx2*np.sin(x5-x2) - 2*dx5*dx3*np.sin(x5-x3) - 2*dx5*dx4*np.sin(x5-x4)

    dl1 =                                 8*dx2*(dx1-dx2)*np.sin(x1-x2) + 6*dx3*(dx1-dx3)*np.sin(x1-x3) + 4*dx4*(dx1-dx4)*np.sin(x1-x4) + 2*dx5*(dx1-dx5)*np.sin(x1-x5)
    dl2 = 8*dx1*(dx2-dx1)*np.sin(x2-x1) +                               + 6*dx3*(dx2-dx3)*np.sin(x2-x3) + 4*dx4*(dx2-dx4)*np.sin(x2-x4) + 2*dx5*(dx2-dx5)*np.sin(x2-x5)
    dl3 = 6*dx1*(dx3-dx1)*np.sin(x3-x1) + 6*dx2*(dx3-dx2)*np.sin(x3-x2) +                               + 4*dx4*(dx3-dx4)*np.sin(x3-x4) + 2*dx5*(dx3-dx5)*np.sin(x3-x5)
    dl4 = 4*dx1*(dx4-dx1)*np.sin(x4-x1) + 4*dx2*(dx4-dx2)*np.sin(x4-x2) + 4*dx3*(dx4-dx3)*np.sin(x4-x3)                                 + 2*dx5*(dx4-dx5)*np.sin(x4-x5)
    dl5 = 2*dx1*(dx5-dx1)*np.sin(x5-x1) + 2*dx2*(dx5-dx2)*np.sin(x5-x2) + 2*dx3*(dx5-dx3)*np.sin(x5-x3) + 2*dx4*(dx5-dx4)*np.sin(x5-x4)    



    Barray =  np.array([[T/l + m*l**2/2 *(dl1 * l1)],
                        [      m*l**2/2 *(dl2 * l2)],
                        [      m*l**2/2 *(dl3 * l3)],
                        [      m*l**2/2 *(dl4 * l4)],
                        [      m*l**2/2 *(dl5 * l5)]])
    
    return Barray






def f1(x1,x2,x3,x4,x5,dx1,dx2,dx3, dx4, dx5,T):
    return (np.linalg.inv(A(x1,x2,x3,x4,x5))@B(x1,x2,x3,x4,x5,dx1,dx2,dx3,dx4,dx5,T))[0][0]
def f2(x1,x2,x3,x4,x5,dx1,dx2,dx3, dx4, dx5,T):
    return (np.linalg.inv(A(x1,x2,x3,x4,x5))@B(x1,x2,x3,x4,x5,dx1,dx2,dx3,dx4,dx5,T))[1][0]
def f3(x1,x2,x3,x4,x5,dx1,dx2,dx3, dx4, dx5,T):
    return (np.linalg.inv(A(x1,x2,x3,x4,x5))@B(x1,x2,x3,x4,x5,dx1,dx2,dx3,dx4,dx5,T))[2][0]
def f4(x1,x2,x3,x4,x5,dx1,dx2,dx3, dx4, dx5,T):
    return (np.linalg.inv(A(x1,x2,x3,x4,x5))@B(x1,x2,x3,x4,x5,dx1,dx2,dx3,dx4,dx5,T))[3][0]
def f5(x1,x2,x3,x4,x5,dx1,dx2,dx3, dx4, dx5,T):
    return (np.linalg.inv(A(x1,x2,x3,x4,x5))@B(x1,x2,x3,x4,x5,dx1,dx2,dx3,dx4,dx5,T))[4][0]



print('hogehoge')
print(f1(0,0,0,0,0,0,0,0,0,0,T))



theta = [[0],
         [0],
         [0],
         [0],
         [0]]
omega = [[0],
         [0],
         [0],
         [0],
         [0]]

x1 = [l*np.cos(theta[0][0])]
x2 = [x1[0] + l*np.cos(theta[1][0])]
x3 = [x2[0] + l*np.cos(theta[2][0])]
x4 = [x3[0] + l*np.cos(theta[3][0])]
x5 = [x4[0] + l*np.cos(theta[4][0])]
y1 = [l*np.sin(theta[0][0])]
y2 = [y1[0] + l*np.sin(theta[1][0])]
y3 = [y2[0] + l*np.sin(theta[2][0])]
y4 = [y3[0] + l*np.sin(theta[3][0])]
y5 = [y4[0] + l*np.sin(theta[4][0])]



for i in range(N):

    
    

    th1_n = theta[0][i]
    th2_n = theta[1][i]
    th3_n = theta[2][i]
    th4_n = theta[3][i]
    th5_n = theta[4][i]
    om1_n = omega[0][i]
    om2_n = omega[1][i]
    om3_n = omega[2][i]
    om4_n = omega[3][i]
    om5_n = omega[4][i]

    T  = 10/N*i
    if i > N/5:
        T = 0


    dk1 = h*f1(th1_n,th2_n,th3_n,th4_n,th5_n, om1_n,om2_n,om3_n, om4_n, om5_n, T)
    dl1 = h*f2(th1_n,th2_n,th3_n,th4_n,th5_n, om1_n,om2_n,om3_n, om4_n, om5_n, T)
    dm1 = h*f3(th1_n,th2_n,th3_n,th4_n,th5_n, om1_n,om2_n,om3_n, om4_n, om5_n, T)
    dn1 = h*f4(th1_n,th2_n,th3_n,th4_n,th5_n, om1_n,om2_n,om3_n, om4_n, om5_n, T)
    do1 = h*f5(th1_n,th2_n,th3_n,th4_n,th5_n, om1_n,om2_n,om3_n, om4_n, om5_n, T)
    
    k1 = h*om1_n
    l1 = h*om2_n
    m1 = h*om3_n
    n1 = h*om4_n
    o1 = h*om5_n

    #print(th1_n + k1/2)
    #print(th2_n + l1/2)
    #print(th3_n + m1/2)
    #print(th4_n + n1/2)
    #print(th5_n + o1/2)
    #print(om1_n + dk1/2)
    #print(om2_n + dl1/2)
    #print(om3_n + dm1/2)
    #print(om4_n + dn1/2)
    #print(om5_n + do1/2)

    dk2 = h*f1(th1_n + k1/2, th2_n + l1/2, th3_n + m1/2, th4_n + n1/2, th5_n + o1/2, om1_n + dk1/2, om2_n + dl1/2, om3_n + dm1/2, om4_n + dn1/2, om5_n + do1/2, T)
    dl2 = h*f2(th1_n + k1/2, th2_n + l1/2, th3_n + m1/2, th4_n + n1/2, th5_n + o1/2, om1_n + dk1/2, om2_n + dl1/2, om3_n + dm1/2, om4_n + dn1/2, om5_n + do1/2, T)
    dm2 = h*f3(th1_n + k1/2, th2_n + l1/2, th3_n + m1/2, th4_n + n1/2, th5_n + o1/2, om1_n + dk1/2, om2_n + dl1/2, om3_n + dm1/2, om4_n + dn1/2, om5_n + do1/2, T)
    dn2 = h*f4(th1_n + k1/2, th2_n + l1/2, th3_n + m1/2, th4_n + n1/2, th5_n + o1/2, om1_n + dk1/2, om2_n + dl1/2, om3_n + dm1/2, om4_n + dn1/2, om5_n + do1/2, T)
    do2 = h*f5(th1_n + k1/2, th2_n + l1/2, th3_n + m1/2, th4_n + n1/2, th5_n + o1/2, om1_n + dk1/2, om2_n + dl1/2, om3_n + dm1/2, om4_n + dn1/2, om5_n + do1/2, T)

    k2 = h*(om1_n + dk1/2)
    l2 = h*(om2_n + dl1/2)
    m2 = h*(om3_n + dm1/2)
    n2 = h*(om4_n + dn1/2)
    o2 = h*(om5_n + do1/2)

    dk3 = h*f1(th1_n + k2/2, th2_n + l2/2, th3_n + m2/2, th4_n + n2/2, th5_n + o2/2, om1_n + dk2/2, om2_n + dl2/2, om3_n + dm2/2, om4_n + dn2/2, om5_n + do2/2, T)
    dl3 = h*f2(th1_n + k2/2, th2_n + l2/2, th3_n + m2/2, th4_n + n2/2, th5_n + o2/2, om1_n + dk2/2, om2_n + dl2/2, om3_n + dm2/2, om4_n + dn2/2, om5_n + do2/2, T)
    dm3 = h*f3(th1_n + k2/2, th2_n + l2/2, th3_n + m2/2, th4_n + n2/2, th5_n + o2/2, om1_n + dk2/2, om2_n + dl2/2, om3_n + dm2/2, om4_n + dn2/2, om5_n + do2/2, T)
    dn3 = h*f4(th1_n + k2/2, th2_n + l2/2, th3_n + m2/2, th4_n + n2/2, th5_n + o2/2, om1_n + dk2/2, om2_n + dl2/2, om3_n + dm2/2, om4_n + dn2/2, om5_n + do2/2, T)
    do3 = h*f5(th1_n + k2/2, th2_n + l2/2, th3_n + m2/2, th4_n + n2/2, th5_n + o2/2, om1_n + dk2/2, om2_n + dl2/2, om3_n + dm2/2, om4_n + dn2/2, om5_n + do2/2, T)

    k3 = h*(om1_n + dk2/2)
    l3 = h*(om2_n + dl2/2)
    m3 = h*(om3_n + dm2/2)
    n3 = h*(om4_n + dn2/2)
    o3 = h*(om5_n + do2/2)

    dk4 = h*f1(th1_n + k3, th2_n + l3, th3_n + m3, th4_n + n3, th5_n + o3, om1_n + dk3, om2_n + dl3, om3_n + dm3, om4_n + dn3, om5_n + do3, T)
    dl4 = h*f2(th1_n + k3, th2_n + l3, th3_n + m3, th4_n + n3, th5_n + o3, om1_n + dk3, om2_n + dl3, om3_n + dm3, om4_n + dn3, om5_n + do3, T)
    dm4 = h*f3(th1_n + k3, th2_n + l3, th3_n + m3, th4_n + n3, th5_n + o3, om1_n + dk3, om2_n + dl3, om3_n + dm3, om4_n + dn3, om5_n + do3, T)
    dn4 = h*f4(th1_n + k3, th2_n + l3, th3_n + m3, th4_n + n3, th5_n + o3, om1_n + dk3, om2_n + dl3, om3_n + dm3, om4_n + dn3, om5_n + do3, T)
    do4 = h*f5(th1_n + k3, th2_n + l3, th3_n + m3, th4_n + n3, th5_n + o3, om1_n + dk3, om2_n + dl3, om3_n + dm3, om4_n + dn3, om5_n + do3, T)

    k4 = h*(om1_n + dk3)
    l4 = h*(om2_n + dl3)
    m4 = h*(om3_n + dm3)
    n4 = h*(om4_n + dn3)
    o4 = h*(om5_n + do3)

    om1_n_1 = om1_n + (dk1 + 2*dk2 + 2*dk3 + dk4)/6
    om2_n_1 = om2_n + (dl1 + 2*dl2 + 2*dl3 + dl4)/6
    om3_n_1 = om3_n + (dm1 + 2*dm2 + 2*dm3 + dm4)/6
    om4_n_1 = om4_n + (dn1 + 2*dn2 + 2*dn3 + dn4)/6
    om5_n_1 = om5_n + (do1 + 2*do2 + 2*do3 + do4)/6

    th1_n_1 = th1_n + (k1 + 2*k2 + 2*k3 + k4)/6
    th2_n_1 = th2_n + (l1 + 2*l2 + 2*l3 + l4)/6
    th3_n_1 = th3_n + (m1 + 2*m2 + 2*m3 + m4)/6
    th4_n_1 = th4_n + (n1 + 2*n2 + 2*n3 + n4)/6
    th5_n_1 = th5_n + (o1 + 2*o2 + 2*o3 + o4)/6

    omega[0].append(om1_n_1)
    omega[1].append(om2_n_1)
    omega[2].append(om3_n_1)
    omega[3].append(om4_n_1)
    omega[4].append(om5_n_1)

    theta[0].append(th1_n_1)
    theta[1].append(th2_n_1)
    theta[2].append(th3_n_1)
    theta[3].append(th4_n_1)
    theta[4].append(th5_n_1)

    print('checker')
    print(i)

    x1.append(l*np.cos(th1_n_1))
    y1.append(l*np.sin(th1_n_1))
    x2.append(x1[i+1] + l*np.cos(th2_n_1))
    y2.append(y1[i+1] + l*np.sin(th2_n_1))
    x3.append(x2[i+1] + l*np.cos(th3_n_1))
    y3.append(y2[i+1] + l*np.sin(th3_n_1))
    x4.append(x3[i+1] + l*np.cos(th4_n_1))
    y4.append(y3[i+1] + l*np.sin(th4_n_1))
    x5.append(x4[i+1] + l*np.cos(th5_n_1))
    y5.append(y4[i+1] + l*np.sin(th5_n_1))


print(x1)


fig = plt.figure()
ax = fig.add_subplot(111)

line, = ax.plot([], [], 'o-', linewidth=100,c = 'red',animated = True)

def update(f):
    ax.cla() # ax をクリア
    ax.grid()
    ax.set_xlim(-(5.5*l), 5.5*l)
    ax.set_ylim(-(5.5*l), 5.5*l)
    ax.set_aspect('equal')
    ax.grid()
    ax.plot(0,0,"o-", c="red")
    ax.plot(x1[int(f)],y1[int(f)],"o-", c="red")
    ax.plot(x2[int(f)],y2[int(f)],"o-", c="purple")
    ax.plot(x3[int(f)],y3[int(f)],"o-", c="blue")
    ax.plot(x4[int(f)],y4[int(f)],"o-", c="green")
    ax.plot(x5[int(f)],y5[int(f)],"o-", c="yellow")
    line.set_data([0, x1[int(f)], x2[int(f)], x3[int(f)], x4[int(f)], x5[int(f)]], [0, y1[int(f)], y2[int(f)], y3[int(f)], y4[int(f)], y5[int(f)]])
    

    #ax.plot(np.cos(f), np.sin(f), "o", c="red")

anim = FuncAnimation(fig, update, frames=range(N), interval=20)

anim.save("c03.gif")#, writer="Pillow")
plt.show()
#plt.close()