import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
# Importing our function from odesolvers.py
from odesolvers import rkf45, rk4
import time

start=time.time()


def dxdt(t, x, v, mu):
    return v, -2*zeta*w0*v-w0**2*x + A*np.sin(omega*t)

def arctan(x):
    return 1/np.tan(x)

# Initial conditions:
t0 = 0
x0 = 1
v0 = 0
w0 = 1
A = 1
omega = 2.5
zeta = 0.07
t1 = 80

# Call adaptive RKF45 method.
function = rkf45(dxdt, t0, t1, x0, v0, A, omega, 'data/task4_1.txt', 'data/task4_2.txt')

# Read data from file, to Pandas dataframe.
df = pd.read_csv('data/task4_1.txt', header=None)

# Get; x and t from dataframe.
x = df.iloc[::2,:]
t = df.iloc[1::2,:]

# Plots
fig, ax1 = plt.subplots(1, 1)
#ax1.plot(t, x, '-', label ='RKF45')
ax1.plot(t, x, '--', color='black', alpha=1)
ax1.set(title='RKF45 adaptive step size', xlabel='time(s)', ylabel='x(t)')
plt.show()
end = time.time()
print((end-start)/60)
