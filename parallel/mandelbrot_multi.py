import numpy as np
from multiprocessing import Pool
import timeit
import matplotlib.pyplot as plt
from numba import jit

def calculate(x):
    iterations = 0
    z = 0
    while abs(z) <= 2 and iterations < max_iterations:
        z = z**2 + x
        iterations += 1
    return iterations


def plot(reStart, reEnd, imStart, imEnd, prc, f):
    pixel_X = np.linspace(reStart, reEnd, 800)
    pixel_Y = np.linspace(imStart, imEnd, 600)

    xx, yy = np.meshgrid(pixel_X, pixel_Y)
    mat = np.array([xx, yy])


    pairs = []
    for i in range(len(mat[0])):
        for j in range(len(mat[0][0])):
            pairs_1 = mat[:, i, j]
            comp = complex(pairs_1[0], pairs_1[1])
            pairs.append(comp)

    pool = Pool(processes=prc)

    n = pool.map(f, pairs)


max_iterations = 1000 # higher = better detail, but more intensive

reStart = -2
reEnd = 1
imStart = -1
imEnd = 1

if __name__ == '__main__':
    prcs = [1,2,3,4,5,6,7,8,9,10]
    ts = []
    tsj = []
    for i in range(len(prcs)):
        start = timeit.default_timer()
        iter_mat = plot(reStart, reEnd, imStart, imEnd, prcs[i], calculate)
        stop = timeit.default_timer()
        ts.append(stop-start)

        start = timeit.default_timer()
        iter_numba = plot(reStart, reEnd, imStart, imEnd, prcs[i], jit(nopython=True)(calculate))
        stop = timeit.default_timer()
        tsj.append(stop-start)
        print(i)

    plt.figure()
    plt.plot(prcs, ts, 'r-', label='Multiprocessing enabled')
    plt.plot(prcs, tsj, 'g-', label='Multiprocessing and Numba enabled')
    plt.plot(1, 20.8494770526886, 'bo', label='Standard Python (no Multiprocessing or Numba)')
    plt.plot(1, 1.2725188732147217, 'bx', label='Standard Python with Numba (no Multiprocessing)')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (seconds)')
    plt.title('Time to Calculate the Mandelbrot Set vs No. Processes (6 Core CPU)')
    plt.legend()
    plt.show()
