import numpy as np
from numpy.linalg import norm
import time
from sklearn.decomposition import NMF

np.random.seed()
tt = 0
nmf = NMF(n_components=128, init='random', random_state=0, solver='mu')
for i in range(2):
    print(i)
    a = np.random.sample([128, 320000])
    since = time.time()
    W = nmf.fit_transform(a)
    H = nmf.components_
    tt +=time.time() - since

print(tt/2)