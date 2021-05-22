import matplotlib.pyplot as plt

data = [20.58795690536499,
11.658324956893921,
8.593661785125732,
7.043179035186768,
6.249681234359741,
5.665907859802246,
6.061463117599487,
6.146204948425293,
6.280256986618042,
6.858752012252808]

prcs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.figure()
plt.plot(prcs, data, 'r-', label='multiprocessing enabled')
plt.plot(1, 20.8494770526886, 'bo', label='no multiprocessing')
plt.xlabel('Number of Processes')
plt.ylabel('Time (seconds)')
plt.title('Time to Calculate the Mandelbrot Set vs No. Processes (6 Core CPU)')
plt.legend()
plt.show()
