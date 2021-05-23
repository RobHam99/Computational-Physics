import numpy as np
import matplotlib.pyplot as plt
from algorithms import velocity_verlet


def f(x):
    """
    Central Force
    """
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    return -initial_alpha(x0)*x/r**3


def initial_alpha(x0):
    """
    Initialise the constant alpha to pass to function 'f'

    This isnt needed in the case of task 2 since r = 1 but i was told to add it in anyway in the case
    of different intial coordinates.
    """
    return w**2 * np.sqrt(x0[0]**2 + x0[1]**2 + x0[2]**2)**3


# Initial conditions
w = 2*np.pi
m = 1
x0 = [-1, 0, 0]
v0 = [0, 2*np.pi, 0]
h = 1E-4 # delta t

# x is an array of length=len(t), containing (x_i, y_i, z_i)
# v is the same as x but for velocities
t = np.arange(0, 10, h)
x = np.zeros((len(t), 3))
v = np.zeros((len(t), 3))
delX = np.zeros((len(t), 3))

x[0] = x0
v[0] = v0

# Call function from algorithms.py
x, v = velocity_verlet(f, x, v, t, h)

# Split x vector into x,y,z coords
x_coords = [i[0] for i in x]
y_coords = [i[1] for i in x]
z_coords = [i[2] for i in x]

def ang(x_p, v_p, m_p):
    array = []
    for i in range(len(x_p)):
        r = np.sqrt((x_p[i][0])**2 + (x_p[i][1])**2 + (x_p[i][2])**2)
        v = np.sqrt((v_p[i][0])**2 + (v_p[i][1])**2 + (v_p[i][2])**2)
        array.append(m_p * v * r)
    return array


angmom = ang(x, v, m)


# Plot y vs x
# Plot y vs x
fig, ((ax1, ax2)) = plt.subplots(1, 2)
ax1.plot(x_coords, y_coords, 'b')
ax1.set(xlabel='x', ylabel='y')

ax2.plot(t, angmom)
ax2.set(xlabel='t', ylabel='L')
#ax2.set_ylim([6, 7])
plt.show()
