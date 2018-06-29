import tensorflow as tf
import numpy as np

a = np.random.sample([50, 60])
b = np.random.sample([50, 60])

import matplotlib.pyplot as plt

figs, axs = plt.subplots(1, 2, figsize=(20, 5))
axs[0].set_aspect('equal')
axs[0].imshow(a, interpolation='nearest', cmap=plt.cm.ocean)
axs[1].set_aspect('equal')
axs[1].imshow(b, interpolation='nearest', cmap=plt.cm.ocean)
#plt.colorbar()
plt.show()