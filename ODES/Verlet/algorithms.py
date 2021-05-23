import numpy as np


def basic_verlet(f, x0, v0, x, v, t, h, delX):
    """
    Basic Verlet algorithm using Stoermer's rule

    -- UNUSED IN THE PROJECT --
    """
    for i in range(len(t)-1):

        if i == 0:
            delX[0] = (v[0]+(h/2)*f(x[0]))*h
            x[1] = x[0] + delX[0]

        else:
            delX[i] = delX[i-1] + f(x[i])*h**2

            x[i+1] = x[i] + delX[i]

            v[i+1] = delX[i]/h + f(x[i+1])*h/2

    return x, v


def velocity_verlet(f, x, v, t, h):
    """
    Velocity Verlet algorithm
    """
    a = f(x[0])
    for i in range(len(x)-1):

        v_half = v[i] + a * h/2

        x[i+1] = x[i] + h * v_half

        a = f(x[i+1])
        v[i+1] = v_half + a * h/2

        print(i*h, end='\r')

    return x, v


def two_body_verlet(f, xe0, ve0, xs0, vs0, t0, t1, h, filename):
    """
    Velocity Verlet adapted for a two body system
    """
    fi = open(f'{filename}', 'w')

    t = t0
    xe = xe0
    xs = xs0
    ve = ve0
    vs = vs0
    fi.write(f'{t},{xe[0]},{xe[1]},{xe[2]},{ve[0]},{ve[1]},{ve[2]},{xs[0]},{xs[1]},{xs[2]},{vs[0]},{vs[1]},{vs[2]}\n')
    ae = f(xe, xs, True)
    asun = f(xe, xs, False)
    while t < t1:

        # half steps
        ve_half = ve + ae * h/2
        vs_half = vs + asun * h/2

        # full position steps
        xe_new = xe + h * ve_half
        xs_new = xs + h * vs_half

        # accelerations at [i+1]
        ae = f(xe_new, xs_new, True)
        asun = f(xe_new, xs_new, False)

        # full velocity steps
        ve_new = ve_half + ae * h/2
        vs_new = vs_half + asun * h/2

        # update variables
        t += h
        xe = xe_new
        ve = ve_new
        xs = xs_new
        vs = vs_new
        fi.write(f'{t},{xe[0]},{xe[1]},{xe[2]},{ve[0]},{ve[1]},{ve[2]},{xs[0]},{xs[1]},{xs[2]},{vs[0]},{vs[1]},{vs[2]}\n')

        print(t, end='\r')


def multi_body_verlet(f, x_in, v_in, t0, t1, h, masses, filename):
    """
    Velocity Verlet adapted for a multiple body system
    """
    fi = open(f'{filename}', 'w')

    t = t0

    # write initial time and each component of each position and velocity to a csv file
    fi.write(f'{t},')
    for i in range(len(x_in)):
        for j in range(len(x_in[0])):
            fi.write(f'{x_in[i][j]},')
        for j in range(len(v_in[0])):
            if i == len(x_in)-1 and j == len(x_in[0])-1:
                fi.write(f'{v_in[i][j]}')
            else:
                fi.write(f'{v_in[i][j]},')
    fi.write('\n')

    # x and v new are the i+1 stages
    x_new = np.zeros((len(x_in), 3))
    v_new = np.zeros((len(v_in), 3))
    v_half = np.zeros((len(v_in), 3))
    a = f(x_in, masses)
    while t < t1:

        # calculate x new for each mass)
        v_half = v_in + a*h*0.5
        x_new = x_in + h*v_half

        # calculate v new for each mass using a[i+1]
        a = f(x_new, masses)
        v_new = v_half + a*h*0.5

        # update variables
        t += h
        x_in = x_new
        v_in = v_new

        # same as for initial write
        fi.write(f'{t},')
        for i in range(len(x_in)):
            for j in range(len(x_in[0])):
                fi.write(f'{x_in[i][j]},')
            for j in range(len(v_in[0])):
                if i == len(x_in)-1 and j == len(x_in[0])-1:
                    fi.write(f'{v_in[i][j]}')
                else:
                    fi.write(f'{v_in[i][j]},')
        fi.write('\n')

        # view current time in program in console
        print(t, end='\r')


def multi_body_verlet_boundary(f, box, x_in, v_in, t0, t1, h, masses, L, filename1):
    """
    Velocity Verlet adapted for a multiple body system with periodic boundarys and minimum
    image convention.
    """

    t = t0
    # x and v new are the i+1 stages
    x_new = np.zeros((len(x_in), 3))
    v_new = np.zeros((len(v_in), 3))
    v_half = np.zeros((len(x_in), 3))
    ix = np.zeros((len(x_in), 3))

    fi = open(f'{filename1}', 'w')

    # write initial time and each component of each position and velocity to a csv file
    fi.write(f'{t},')
    for i in range(len(x_in)):
        for j in range(len(x_in[0])):
            fi.write(f'{x_in[i][j]},')
        for j in range(len(v_in[0])):
            if i == len(x_in)-1 and j == len(x_in[0])-1:
                fi.write(f'{v_in[i][j]}')
            else:
                fi.write(f'{v_in[i][j]},')
    fi.write('\n')

    a = f(x_in, masses, v_in, h)
    while t < t1:

        # verlet integration for x[i+1]
        v_half = v_in + a*h/2
        x_new = x_in + h*v_half

        # verlet integration for v[i+1]
        a = f(x_new, masses, v_in, h)
        v_new = v_half + a*h/2

        # check for particles outside of box at x[i+1]
        x_new, ix = box(x_new, ix)

        # update variables
        t += h
        x_in = x_new
        v_in = v_new
        # same as for initial write
        fi.write(f'{t},')
        for i in range(len(x_in)):
            for j in range(len(x_in[0])):
                fi.write(f'{x_in[i][j]+ix[i][j]*L[j]},')
            for j in range(len(v_in[0])):
                if i == len(x_in)-1 and j == len(x_in[0])-1:
                    fi.write(f'{v_in[i][j]}')
                else:
                    fi.write(f'{v_in[i][j]},')
        fi.write('\n')

        # view current time in program in console
        print(t, end='\r')
