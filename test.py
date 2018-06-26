import numpy as np
from numpy.linalg import norm

from sklearn.decomposition import NMF


nmf = NMF(n_components=20, init='random', random_state=0, solver='mu')
a = np.random.sample((128, 320000))
b = np.random.sample((16, 4))
wa, ha = nmf.fit_transform(a), nmf.components_
wb, hb = nmf.fit_transform(b), nmf.components_
print(norm(np.matmul(wa, ha) - a), norm(np.matmul(wb, hb) - b))
print(ha.shape, hb.shape)