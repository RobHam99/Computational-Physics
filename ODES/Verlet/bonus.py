import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
# uncomment if you ran new trajectories in the task 8 file and want
# to calculate the MSD of them instead of the presaved ones from the report
df = pd.read_csv('data/task_8.csv', header=None)

#df.to_csv('data/bonus_epsilon3.csv')

# df = pd.read_csv('data/bonus_epsilon0.csv')
# first and last lines of the data file
# i.e. t0 and tmax
first = df.iloc[0,:]
last = df.iloc[-1,:]

Natom = 50

# initialise arrays and counters
r0 = np.zeros((Natom, 3))
rt = np.zeros((Natom, 3))
j = 0
i = 0
k = 0
while i < len(first):
    # pick out the positions since they are sandwiched between the
    # velocities, skip index 0 since its time, take the first 3
    # next indices since they are positions, then do nothing
    # for the next 3 indices since they are velocities
    # then repeat the process
    if j == 1:
        k += 1
    if j == 1 or j == 2 or j == 3:
        r0[k-1][j-1] = first[i]
        rt[k-1][j-1] = last[i]
    if j == 6:
        j = 0

    j += 1
    i += 1

msd_diff = rt-r0

# calculate each magnitude of rt-r0
msd_mag = np.zeros((Natom))
for i in range(len(msd_diff)):
    msd_mag[i] = np.sqrt(msd_diff[i][0]**2+msd_diff[i][1]**2+msd_diff[i][2]**2)

# final msd
msd = sum(msd_mag**2)/Natom
print(msd)

# write msd arrays to txt file, when bringing them out you need
# to wrap them in float(value) or they will be a string and not work
#fi = open('data/msd_epsi3.txt', 'w')
#for i in msd_mag:
#    fi.write(f'{i**2}\n')
#"""

# msd expectation value for comparison
t = 20
real_msd = 6*t/25

# this code plots our presaved MSDs
msd_plot = []
sum_msd = 0
j = 0
fi = open('data/msd_epsi2.txt', 'r')
for i in fi:
    msd_plot.append(float(i))
    j += 1
    sum_msd += float(i)

av_msd = sum_msd/j
print(av_msd, '\n', real_msd)


tlin = np.linspace(0, t, len(msd_plot))
real = np.full((len(msd_plot)), real_msd)
plt.figure()
plt.plot(tlin, msd_plot, 'r-', label='MSD(t)')
plt.plot(tlin, real, 'b-', label='$MSD_{const}$')
plt.xlabel('t')
plt.ylabel('MSD')
plt.legend()
plt.show()
