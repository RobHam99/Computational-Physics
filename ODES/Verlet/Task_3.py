import numpy as np
import matplotlib.pyplot as plt
from algorithms import two_body_verlet
import pandas as pd


def f(xe, xs, orbit):
    """
#    Central Force, the boolean is just to decided whether to return earth or sun acceleration
#    will figure out a cleaner way to do this at some point
#    """
    p = xe-xs
    r = np.sqrt((xe[0]-xs[0])**2+(xe[1]-xs[1])**2+(xe[2]-xs[2])**2)
    F_12 = -G*m1*m2*p/r**3
    if orbit is True:
        return F_12/m1

    if orbit is False:
        return -F_12/m2


def ang(x_p, v_p, m_p):
    """
    Calculates angular momentum L=mvr
    """
    array = []
    for i in range(len(x_p)):
        r = np.sqrt((x_p[i][0])**2 + (x_p[i][1])**2 + (x_p[i][2])**2)
        v = np.sqrt((v_p[i][0])**2 + (v_p[i][1])**2 + (v_p[i][2])**2)
        array.append(m_p * v * r)
    return array


# Initial conditions
m1 = 1
m2 = 332959
G = 1.185684E-4

# Initial conditions Earth
xe0 = np.array([1.0168, 0, 0])
ve0 = np.array([0, 6.1786, 0])

# Initial conditions Sun
xs0 = np.array([-3.05383E-6, 0, 0])
vs0 = np.array([0, -1.85512E-5, 0])

t0 = 0
t1 = 10.
h = 1E-4 # delta t

# Call function from algorithms.py
func = two_body_verlet(f, xe0, ve0, xs0, vs0, t0, t1, h, 'data/task_3.csv')

df = pd.read_csv('data/task_3.csv', header=None)
t = df.iloc[:,0].to_numpy()
xe = df.iloc[:,1:4].to_numpy()
ve = df.iloc[:,4:7].to_numpy()

angmom = ang(xe, ve, m1)

# Plot y vs x
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(df[1], df[2], 'b', df[7], df[8], 'r')
ax1.set(xlabel='x', ylabel='y(x)')

ax2.plot(t, df[1])
ax2.set(xlabel='t', ylabel='x(t)')

ax3.plot(t, df[2])
ax3.set(xlabel='t', ylabel='y(t)')

ax4.plot(t, angmom)
ax4.set(xlabel='t', ylabel='L(t)')
plt.show()
