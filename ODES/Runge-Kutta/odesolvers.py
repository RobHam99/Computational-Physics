import numpy as np
import math
import matplotlib.pyplot as plt
import copy

def rk2(f, start, stop, x0, v0, w0, A, zeta, omega, step):
    '''
    Runge-Kutta solver, order 2.
    '''

    # Initial setup
    t_grid = np.arange(start, stop, step)

    x = np.zeros(len(t_grid))
    v = x.copy()
    v[0] = v0
    x[0] = x0

    # Loop through specified steps
    for i in range(len(x) - 1):

        # k is the first equation of the system of ODEs, l is the second.

        k1 = step * f(t_grid[i], x[i], v[i])[0]
        l1 = step * f(t_grid[i], x[i], v[i])[1]

        k2 = step * f(t_grid[i] + step/2, x[i] + k1/2, v[i] + l1/2)[0]
        l2 = step * f(t_grid[i] + step/2, x[i] + k1/2, v[i] + l1/2)[1]

        # k vals give x, l vals give v
        x[i+1] = x[i] + k2
        v[i+1] = v[i] + l2

    return x, t_grid, v


def rk4(f, start, stop, x0, v0, w0, A, zeta, omega, step):
    '''
    Runge-Kutta solver, order 4.
    '''

    # Initial setup
    t_grid = np.arange(start, stop, step)

    x = np.zeros(len(t_grid))
    v = x.copy()
    v[0] = v0
    x[0] = x0

    # Loop through specified steps
    for i in range(len(x) - 1):

        # k is the first equation of the system of ODEs, l is the second.
        k1 = step * f(t_grid[i], x[i], v[i])[0]
        l1 = step * f(t_grid[i], x[i], v[i])[1]

        k2 = step * f(t_grid[i] + step/2, x[i] + k1/2, v[i] + l1/2)[0]
        l2 = step * f(t_grid[i] + step/2, x[i] + k1/2, v[i] + l1/2)[1]

        k3 = step * f(t_grid[i] + step/2, x[i] + k2/2, v[i] + l2/2)[0]
        l3 = step * f(t_grid[i] + step/2, x[i] + k2/2, v[i] + l2/2)[1]

        k4 = step * f(t_grid[i] + step, x[i] + k3, v[i] + l3)[0]
        l4 = step * f(t_grid[i] + step, x[i] + k3, v[i] + l3)[1]

        # k vals give x, l vals give v
        x[i+1] = x[i] + k1/6 + k2/3 + k3/3 + k4/6
        v[i+1] = v[i] + l1/6 + l2/3 + l3/3 + l4/6

    return x, t_grid, v


def rkf45(f, t0, t1, x0, v0, A, omega, filename1, filename2, mu=0):
    """
    Adaptive Runge-Kutta Fehlberg algorithm.
    """

    # Initial conditions
    h = 0.01
    t = t0
    x = x0
    v = v0
    lam = 0.9
    e0 = 1E-9

    # Using name 'fi' if you use standard 'f' then it tries to call the function dxdt

    fi = open(f'{filename1}', 'w')
    fi2 = open(f'{filename2}', 'w')

    # Butcher tableau coefficients for beta and alpha (only the ones that are fractions, for better speed)
    c0 = 1/4
    c1 = 3.0/8.0
    c2 = 3.0/32.0
    c3 = 9.0/32.0
    c4 = 12.0/13.0
    c5 = 1932.0/2197.0
    c6 = 7200.0/2197.0
    c7 = 7296.0/2197.0
    c8 = 439.0/216.0
    c9 = 3680.0/513.0
    c10 = 845.0/4104.0
    c11 = 0.5
    c12 = 8.0/27.0
    c13 = 3544.0/2565.0
    c14 = 1859.0/4104.0
    c15 = 11.0/40.0

    # Coefficients to calculate error and delta x and delta v
    ch = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0])
    ck = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0])
    ct = ck-ch

    # k is the vector for the first ODE, l is for the second.
    k = np.zeros(6)
    l = np.zeros(6)

    while t <= t1:

        # Compute each k and l value, the order of this matters.
        k[0] = h * f(t, x, v, mu)[0]
        l[0] = h * f(t, x, v, mu)[1]

        k[1] = h * f(t + c0*h, x + c0*k[0], v + c0*l[0], mu)[0]
        l[1] = h * f(t + c0*h, x + c0*k[0], v + c0*l[0], mu)[1]

        k[2] = h * f(t + c1*h, x + c2*k[0] + c3*k[1], v + c2*l[0] + c3*l[1], mu)[0]
        l[2] = h * f(t + c1*h, x + c2*k[0] + c3*k[1], v + c2*l[0] + c3*l[1], mu)[1]

        k[3] = h * f(t + c4*h, x + c5*k[0] - c6*k[1] + c7*k[2], v + c5*l[0] - c6*l[1] + c7*l[2], mu)[0]
        l[3] = h * f(t + c4*h, x + c5*k[0] - c6*k[1] + c7*k[2], v + c5*l[0] - c6*l[1] + c7*l[2], mu)[1]

        k[4] = h * f(t + h, x + c8*k[0] - 8.0*k[1] + c9*k[2] - c10*k[3], v + c8*l[0] - 8.0*l[1] + c9*l[2] - c10*l[3], mu)[0]
        l[4] = h * f(t + h, x + c8*k[0] - 8.0*k[1] + c9*k[2] - c10*k[3], v + c8*l[0] - 8.0*l[1] + c9*l[2] - c10*l[3], mu)[1]

        k[5] = h * f(t + c11*h, x - c12*k[0] + 2.0*k[1] - c13*k[2] + c14*k[3] - c15*k[4], v - c12*l[0] + 2.0*l[1] - c13*l[2] + c14*l[3] - c15*l[4], mu)[0]
        l[5] = h * f(t + c11*h, x - c12*k[0] + 2.0*k[1] - c13*k[2] + c14*k[3] - c15*k[4], v - c12*l[0] + 2.0*l[1] - c13*l[2] + c14*l[3] - c15*l[4], mu)[1]


        # Calculate the error and change in x, and v.
        error = 0.0
        delta_x = 0.0
        delta_v = 0.0
        for i in range(len(ct)):
            error += ct[i]*k[i]
            delta_x += ck[i]*k[i]
            delta_v += ck[i]*l[i]

        # If the error is too large, adjust step size and repeate iteration.
        error = abs(error)
        if error > e0:
            h = lam * h * (e0 / error)**0.2
            continue

        # Compute; x, v, t and write them to file. Then compute optimal step size
        # and move to next loop.
        x += delta_x
        v += delta_v
        t += h
        print('   ', t, end='\r')
        fi.write('%f\n %f\n' % (x, t))
        fi2.write('%f\n %f\n' % (v, t))
        if error <= e0:
            h = lam * h * (e0 / error)**0.2
