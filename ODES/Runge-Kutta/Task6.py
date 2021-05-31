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

# Different mu values for x0 = 1 or x0 = 3
x1_mu0_1 = rkf45(dxdt, t0, 500, x0, v0, A, omega, 'data/task6_1_mu_0_1.txt', 'data/task6_2_mu_0_1.txt', 0.1)
x1_mu0_3 = rkf45(dxdt, t0, 500, x0, v0, A, omega, 'data/task6_1_mu_0_3.txt', 'data/task6_2_mu_0_3.txt', 0.3)
x1_mu0_7 = rkf45(dxdt, t0, 500, x0, v0, A, omega, 'data/task6_1_mu_0_7.txt', 'data/task6_2_mu_0_7.txt', 0.7)
x1_mu1 = rkf45(dxdt, t0, 500, x0, v0, A, omega, 'data/task6_1_mu_1.txt', 'data/task6_2_mu_1.txt', 1.)
x1_mu3 = rkf45(dxdt, t0, 500, x0, v0, A, omega, 'data/task6_1_mu_3.txt', 'data/task6_2_mu_3.txt', 3.)
x1_mu5 = rkf45(dxdt, t0, 500, x0, v0, A, omega, 'data/task6_1_mu_5.txt', 'data/task6_2_mu_5.txt', 5.)

# Datasets
df = pd.read_csv('data/task6_1_mu_0_1.txt', header=None)
df2 = pd.read_csv('data/task6_2_mu_0_1.txt', header=None)

df3 = pd.read_csv('data/task6_1_mu_0_3.txt', header=None)
df4 = pd.read_csv('data/task6_2_mu_0_3.txt', header=None)

df5 = pd.read_csv('data/task6_1_mu_0_7.txt', header=None)
df6 = pd.read_csv('data/task6_2_mu_0_7.txt', header=None)

df7 = pd.read_csv('data/task6_1_mu_1.txt', header=None)
df8 = pd.read_csv('data/task6_2_mu_1.txt', header=None)

df9 = pd.read_csv('data/task6_1_mu_3.txt', header=None)
df10 = pd.read_csv('data/task6_2_mu_3.txt', header=None)

df11 = pd.read_csv('data/task6_1_mu_5.txt', header=None)
df12 = pd.read_csv('data/task6_2_mu_5.txt', header=None)

# Arrays
x = df.iloc[::2,:]
t = df.iloc[1::2,:]
v = df2.iloc[::2,:]

x2 = df3.iloc[::2,:]
t2 = df3.iloc[1::2,:]
v2 = df4.iloc[::2,:]

x3 = df5.iloc[::2,:]
t3 = df5.iloc[1::2,:]
v3 = df6.iloc[::2,:]

x4 = df7.iloc[::2,:]
t4 = df7.iloc[1::2,:]
v4 = df8.iloc[::2,:]

x5 = df9.iloc[::2,:]
t5 = df9.iloc[1::2,:]
v5 = df10.iloc[::2,:]

x6 = df11.iloc[::2,:]
t6 = df11.iloc[1::2,:]
v6 = df12.iloc[::2,:]

# Plots
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
ax1.plot(x, v, 'r-', label = '\u03bc=0.1')
ax1.plot(x2, v2, 'g-', label='\u03bc=0.3')
ax1.plot(x3, v3, 'b-', label='\u03bc=0.7')
ax1.set(title='Phase Portrait', xlabel='x(t) m', ylabel='v(t) (m/s)')
ax1.legend(loc='best')

ax2.plot(x4, v4, 'r-', label = '\u03bc=1')
ax2.plot(x5, v5, 'g-', label='\u03bc=3')
ax2.plot(x6, v6, 'b-', label='\u03bc=5')
ax2.set(title='Phase Portrait', xlabel='x(t) m', ylabel='v(t) (m/s)')
ax2.legend(loc='best')

plt.show()
