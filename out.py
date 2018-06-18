import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import numpy as np

x = np.linspace(0, 10, num=11, endpoint=True)
f = lambda i : i ** 2
y = f(x)
f1 = interp1d(x, y, kind='nearest')

xnew = np.linspace(0, 10, num=1001, endpoint=True)
plt.plot(x, y, 'o')
plt.plot(xnew, f1(xnew), '-', ':')
plt.show()