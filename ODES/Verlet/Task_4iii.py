import numpy as np
import matplotlib.pyplot as plt
from algorithms import multi_body_verlet
import pandas as pd


def f(objects, masses):
    """
    Returns accelerations for each mass x,y,z components
    """
    forces = []
    indices = []
    accelerations = np.zeros((len(masses), 3))

    for i in range(len(masses)):
        for j in range(len(masses)):
            # This part calculates the 3 forces F_12, F_13, F_23
            # also appends the indices of that force
            # so the negative can be used for the opposing mass Fji=-Fij
            if i != j and (j-i)>0:
                # calculates distance vector
                p = objects[i]-objects[j]
                # magnitude of distance vector
                r = np.sqrt(p[0]**2+p[1]**2+p[2]**2)
                # gravitational force
                force = -G*masses[i]*masses[j]*p/r**3
                forces.append(force)
                indices.append([i,j])

                # adds first term of acceleration
                accelerations[i] += force/masses[i]

        # loops through indices
        # if an existing index matches with the current outer loop
        # then it adds the corresponding acceleration
        # e.g. at i = 1: if an index has [0, 1] then the corresponding
        # force from the forces array is used to calculate the acceleration
        # e.g. at i = 1: if an index has [0, 2] then it passes because
        # it only cares about indices with 1 as the second part
        for k in range(len(indices)):
            if indices[k][1] == i:
                accelerations[i] += -forces[k]/masses[i]

    return accelerations

# Initial conditions
m1 = 1
m2 = 332959
m3 = 332959
G = 1.185684E-4

# Initial conditions Earth
x10 = np.array([-6., 0, 0])
v10 = np.array([0, 6., 0])

# Initial conditions Star 1
x20 = np.array([-5., 0, 0])
v20 = np.array([0, 1.5, 0])

# Initial conditions Star 2
x30 = np.array([5., 0, 0])
v30 = np.array([0, -1.5, 0])

t0 = 0.
t1 = 5.
h = 1E-4 # delta t

# Call function from algorithms.py
func = multi_body_verlet(f, [x10, x20, x30], [v10, v20, v30], t0, t1, h, [m1, m2, m3], 'data/task_4iii.csv')

# IF YOUR DATA FILE IS ALREADY FULL, COMMENT OUT THE FUNCTION CALL
# AND RUN THE PROGRAM WITH JUST THE PLOTS TO SAVE LOTS OF TIME

# data frame structure and indices:
# t, x1, y1, z1, v1x, v1y, v1z, x2, y2, z2, v2x, v2y, v2z, x3, y3, z3, v3x, v3y, v3z
# 0   1   2   3    4    5    6   7   8   9   10   11   12  13  14  15   16   17   18

df = pd.read_csv('data/task_4iii.csv', header=None)

fig, ((ax1)) = plt.subplots(1, 1)
ax1.plot(df.iloc[:, 1], df.iloc[:, 2], 'r-', label='x1')
ax1.plot(df.iloc[:, 7], df.iloc[:, 8], 'g-', label='x2')
ax1.plot(df.iloc[:, 13], df.iloc[:, 14], 'b-', label='x3')
ax1.set(xlabel='x', ylabel='y')
plt.legend()
plt.show()
