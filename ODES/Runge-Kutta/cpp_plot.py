import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

t_f = pd.read_csv('t.txt')
x_f = pd.read_csv('x.txt')

print(x_f)

plt.figure()
plt.plot(t_f, x_f)
plt.show()
