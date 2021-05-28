import numpy as np
from numba import jit
from multiprocessing import Pool
import time


@jit(nopython=True)
def calculate(x):
    """
    Receive complex number x, return number of iterations
    that the function stays below 2. Function from:
    https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
    """
    z = 0
    iterations = 0
    while abs(z) <= 2 and iterations < max_iterations:
        z = z**2 + x
        iterations += 1
    return iterations


def plot(reStart, reEnd, imStart, imEnd, iterations):
    """
    Plot the colour for each pixel based on iterations. Adapted from:
    https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
    """
    pixel_X = np.linspace(reStart, reEnd, 800)
    pixel_Y = np.linspace(imStart, imEnd, 600)

    xx, yy = np.meshgrid(pixel_X, pixel_Y)
    mat = np.array([xx, yy])

    if __name__ == '__main__':
        Ns = []
        for i in range(len(mat[0])):
            for j in range(len(mat[0][0])):
                pairs_1 = mat[:, i, j]
                c = complex(pairs_1[0], pairs_1[1])

                n = calculate(c)
                Ns.append(n)



max_iterations = 1000 # higher = better detail, but more intensive

reStart = -2
reEnd = 1
imStart = -1
imEnd = 1
start = time.time()
iter_mat = plot(reStart, reEnd, imStart, imEnd, max_iterations)
stop = time.time()

print(stop-start)
