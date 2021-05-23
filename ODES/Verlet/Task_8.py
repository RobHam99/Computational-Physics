import numpy as np
import matplotlib.pyplot as plt
from algorithms import multi_body_verlet_boundary
import pandas as pd
from mpl_toolkits import mplot3d


def f(objects, masses, velocity, step):
    """
    Returns acceleration for each mass in xyz components
    """
    ljforces = []
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
                    if p[n]>0.5*L[n]:
                        p[n] -= L[n]
                    if p[n]<-0.5*L[n]:
                        p[n] += L[n]
                # magnitude of distance vector
                r = np.sqrt(p[0]**2+p[1]**2+p[2]**2)

                # neglect Lennard-Jones if distance is large enough
                if r <= rc2:
                    ljforce = 4*epsilon*(12*(sigma**12/r**14)-6*(sigma**6/r**8)) * p
                else:
                    ljforce = 0

                ljforces.append(ljforce)
                indices.append([i,j])

                # adds first term of acceleration
                accelerations[i] += ljforce/masses[i]

        # loops through indices
        # if an existing index matches with the current outer loop
        # then it adds the corresponding acceleration
        # e.g. at i = 1: if an index has [0, 1] then the corresponding
        # force from the forces array is used to calculate the acceleration
        # e.g. at i = 1: if an index has [0, 2] then it passes because
        # it only cares about indices with 1 as the second part
        for k in range(len(indices)):
            if indices[k][1] == i:
                accelerations[i] += -ljforces[k]/masses[i]

    # add velocity dependent friction force and stochastic force
    # divided by the particles mass
    for i in range(len(accelerations)):
        fr = np.random.normal(loc=0.0, scale=1.0, size=3)
        accelerations[i] += (-masses[i]*velocity[i]*gamma + np.sqrt((2*masses[i]*gamma*kb*temp)/step)*fr)/masses[i]

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

# Initial conditions
gamma = 25
temp = 1
kb = 1
epsilon = 1
sigma = 1
m = 1
Natom = 6
L = np.array([10, 10, 10])
x = np.zeros((Natom, 3))
v = np.zeros((Natom, 3))
masses = np.full((Natom), m)
rc2 = 2.5
t0 = 0.
t1 = 10.
h = 5E-4 # delta t

# Oliver's code to intialise random positions/velocities and make sure
# they don't overlap
a=0
while a < Natom:
    # initialise random positions with Cartesian components in (-1,+1)
    xp = np.random.uniform(low=-1.0, high=1.0, size=3)
    # scale random coordinates to box size
    xp[0]*=0.5*L[0]
    xp[1]*=0.5*L[1]
    xp[2]*=0.5*L[2]
    # check for overlap, accept pass and reject fail
    for b in range(Natom):
       dr2 = (xp[0]-x[b,0])**2 + (xp[1]-x[b,1])**2 + (xp[2]-x[b,2])**2
       if dr2 < rc2:
           flag = 0
           break
       else:
           flag = 1
    if flag:
       x[a] = xp
       a+=1
    print(f'Particle {a:} initialised\r', end='')
print('\n', end='')
# initialise random velocities with components in (-1,+1)
v = np.random.uniform(low=-1.0, high=1.0, size=(Natom,3))
# normalise and set to thermal velocities
for n in range(Natom):
    vnorm2 = 0

    for i in range(3):
       vnorm2 += v[n,i]**2
    v[n] /= np.sqrt(vnorm2)
    v[n] *= np.sqrt(3*temp/masses[n])

# Call function from algorithms.py
func = multi_body_verlet_boundary(f, box, x, v, t0, t1, h, masses, L, 'data/task_8.csv')

# COMMENT STUFF BELOW OUT IF YOU ARE RUNNING THIS CODE TO USE THE DATA
# FOR THE BONUS FILE SINCE THE STUFF BELOW IS ONLY NECESSARY FOR TASK 8

# data frame structure and indices:
# t, x1, y1, z1, v1x, v1y, v1z, x2, y2, z2, v2x, v2y, v2z, ...
# 0   1   2   3    4    5    6   7   8   9   10   11   12  ...

# data frame
df = pd.read_csv('data/task_8.csv', header=None)


# df.iloc[:200001,column index] for time up to 100 in a t_max=200 plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(df.iloc[:,1], df.iloc[:,2], df.iloc[:,3], 'red', label='p1')
ax.plot3D(df.iloc[:,7], df.iloc[:,8], df.iloc[:,9], 'green', label='p2')
ax.plot3D(df.iloc[:,13], df.iloc[:,14], df.iloc[:,15], 'blue', label='p3')
ax.plot3D(df.iloc[:,19], df.iloc[:,20], df.iloc[:,21], 'black', label='p4')
ax.plot3D(df.iloc[:,25], df.iloc[:,26], df.iloc[:,27], 'aqua', label='p5')
ax.plot3D(df.iloc[:,31], df.iloc[:,32], df.iloc[:,33], 'gray', label='p6')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()
