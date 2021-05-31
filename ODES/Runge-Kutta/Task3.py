import numpy as np
import matplotlib.pyplot as plt
# Importing our functions from odesolvers.py
from odesolvers import rk2, rk4, rkf45
import math
import time

start=time.time()

def dxdt(t, x, v):
    return v, -2*zeta*w0*v-w0**2*x + A*np.sin(omega*t)


def arctan(x):
    return 1/np.tan(x)

# Initial conditions
t0 = 0
x0 = 1
v0 = 0
zeta = 0.07
w0 = 1
A = 1
omega = 2.5
t_a = np.linspace(0, 80, 10000)

# Compute; x, t and v using both RK2 and RK4 methods
x, t, v = rk2(dxdt, t0, 80, x0, v0, w0, A, 0.07, omega, 1)
x2, t2, v2 = rk4(dxdt, t0, 80, x0, v0, w0, A, 0.07, omega, 1)

# Analytical solution for comparison to RK methods.
phi = arctan(2*zeta*w0*omega / w0**2 - omega**2)
M = np.sqrt((w0**2 - omega**2)**2 + (2*zeta*w0*omega)**2)
xp = A*np.sin(omega*t_a - phi) / M

a = -zeta*w0
b = w0*(np.sqrt(1-zeta**2))
lambda1 = a + (1j * b)
lambda2 = a - (1j * b)
k1 = 1 - A*np.sin(-phi)/M
k2 = (-a*k1/b) - (A*omega*np.cos(-phi))/(b*M)

xh = np.exp(a*t_a)*(k1*np.cos(b*t_a) + k2*np.sin(b*t_a))

sol = xh + xp
# Plot; RK2, RK4 and analytical solution.
plt.figure()
plt.scatter(t2, x2, color='blue', label='rk4 with h=1', alpha=0.4)
plt.scatter(t, x, color='purple',label='rk2 with h=1', alpha=0.4)
plt.plot(t_a, sol, '-', color='purple',label='Analytical Solution', alpha=0.6)
plt.legend()
plt.ylabel('x(t)')
plt.ylim(-1.2, 1.2)
plt.title('Runge-Kutta Solvers vs Analytical Solution')
plt.xlabel('time')
plt.show()
print((time.time() - start)/60)
