import numpy as np
import matplotlib.pyplot as plt
from algorithms import multi_body_verlet_boundary
import pandas as pd

def f(objects, masses, velocity, step):
    """
    Returns acceleration for each mass in xyz components
    """
    forces = []
    indices = []
    accelerations = np.zeros((len(masses), 3))

    for i in range(len(masses)):
        for j in range(len(masses)):
            # This part calculates the Lennard-Jones forces for each pair
            # also appends the indices of that force
            # so the negative can be used for the opposing mass Fji=-Fij
            if i != j and (j-i)>0:
                # distance vector
                p = objects[i]-objects[j]
                # Oliver's minimum image convention
                for n in range(len(p)):
                    if p[i]>0.5*L[i]:
                        p[i] -= L[i]
                    if p[i]<-0.5*L[i]:
                        p[i] += L[i]
                # magnitude of distance vector
                r = np.sqrt(p[0]**2+p[1]**2+p[2]**2)
                # Lennard-Jones force
                force = 4*epsilon*(12*(sigma**12/r**13)-6*(sigma**6/r**7)) * p/r

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


def box(x_current, ix_current):
    """
    Function to map particles back to box as per Oliver's code
    """

    for i in range(len(x_current)):
        for j in range(len(x_current[0])):
            if x_current[i][j] < -0.5*L[j]:
                x_current[i][j] += L[j]
                ix_current[i][j] -= 1

            if x_current[i][j] >= 0.5*L[j]:
                x_current[i][j] -= L[j]
                ix_current[i][j] += 1

    return x_current, ix_current


#initial conditions
epsilon = 1
sigma = 1
m = 1
x10 = np.array([-1, -1, -1])
x20 = np.array([1, 1, 1])

v10 = np.array([1, 1, 1])
v20 = np.array([-1, -1, -1])

L = np.array([5, 5, 5])
t0 = 0.
t1 = 100.
h = 1E-4 # delta t

# Call function from algorithms.py
func = multi_body_verlet_boundary(f, box, [x10, x20], [v10, v20], t0, t1, h, [m, m], L, 'data/task_6.csv')

# data frame structure and indices:
# t, x1, y1, z1, v1x, v1y, v1z, x2, y2, z2, v2x, v2y, v2z
# 0   1   2   3    4    5    6   7   8   9   10   11   12

df = pd.read_csv('data/task_6.csv', header=None)

fig, ((ax1)) = plt.subplots(1, 1)
ax1.plot(df[0], df[1], 'r-', label='x1')
ax1.plot(df[0], df[2], 'g-', label='y1')
ax1.plot(df[0], df[3], 'b-', label='z1')
ax1.plot(df[0], df[7], '-', c='blueviolet', label='x2')
ax1.plot(df[0], df[8], '-', c='LawnGreen', label='y2')
ax1.plot(df[0], df[9], '-', c='Magenta', label='z2')
ax1.set(xlabel='t', ylabel='coordinate')
plt.legend()
plt.show()
