import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# Importing our function from odesolvers.py
from odesolvers import rkf45
from numba import jit

# Large datasets -> use this:
mpl.rcParams['agg.path.chunksize']=50001

@jit(nopython=True)
def dxdt(t, x, v, mu):
    '''
    Input function for runge-kutta, 2 first order ODEs for the van der Pol equation with forcing term included.
    '''
    return v, mu*(1-x**2)*v-x + A*np.sin(omega*t)



# Initial conditions:
t0 = 0
x0 = 1.0
v0 = 0
A = 5.0
omega = 3.3706
mu = 5.0
t1 = 50000.0

# Call RKF45 method for van der Pol oscillator
function = rkf45(dxdt, t0, t1, x0, v0, A, omega, 'data/task7_1.txt', 'data/task7_2.txt', mu)

# Datasets
df = pd.read_csv('data/task7_1.txt', header=None)
df2 = pd.read_csv('data/task7_2.txt', header=None)

# Arrays
x = df.iloc[::2,:]
t = df.iloc[1::2,:]
v = df2.iloc[::2,:]

# Plots
plt.figure()
plt.plot(x, v, 'g-', label=rf'$\Omega={omega}$')
plt.title('Phase Portrait')
plt.xlabel('x')
plt.ylabel('v')
plt.legend()
plt.show()
