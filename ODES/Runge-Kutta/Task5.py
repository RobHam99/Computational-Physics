import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# Importing our function from odesolvers.py
from odesolvers import rkf45

def dxdt(t, x, v, mu):
    '''
    Input function for runge-kutta, 2 first order ODEs for the van der Pol equation.
    '''
    return v, mu*(1-x**2)*v-x


# Initial conditions:
t0 = 0
x0 = 3.0
v0 = 0
A = 0
omega = 0
t1 = 4*np.pi
mu = 0

# Call adaptive RKF45 method for the van der Pol oscillator.
function = rkf45(dxdt, t0, t1, x0, v0, A, omega, 'data/task5_1.txt', 'data/task5_2.txt', mu)

# Read file into Pandas dataframe.
df = pd.read_csv('data/task5_1.txt', header=None)
df2 = pd.read_csv('data/task5_2.txt', header=None)

# Read dataframe into arrays for; x, t and v
x = df.iloc[::2,:]
t = df.iloc[1::2,:]
v = df2.iloc[::2,:]

# Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,8), constrained_layout=True)
ax1.plot(t, x, 'k--', label = '\u03bc=0')
ax1.set(title='Trajectory vs Time', xlabel='Time (s)', ylabel='x (m)')
ax1.legend(loc='best')

ax2.plot(x, v, 'b-', label='\u03bc=0')
ax2.set(title='Phase Portrait', xlabel='x (m)', ylabel='v (m/s)')
plt.show()
