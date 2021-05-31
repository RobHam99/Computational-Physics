import numpy as np
import matplotlib.pyplot as plt
# Importing our function from odesolvers.py
from odesolvers import rk2


def dxdt(t, x, v):
    return v, -2*zeta*w0*v-w0**2*x + A*np.sin(omega*t)


# Initial conditions
t0 = 0
x0 = 1
v0 = 0
zeta = 0.07
w0 = 1
A = 1
omega = 2.5

# Compute x, t and v.
x, t, v = rk2(dxdt, t0, 80, x0, v0, w0, A, 0.07, omega, 1e-3)

# Plot
plt.figure()
plt.plot(t, x, '-', label='rk2')
plt.legend()
plt.show()
