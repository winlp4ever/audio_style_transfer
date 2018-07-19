import numpy as np
from scipy.linalg import ldl
from numpy.linalg import norm, eigh

a = np.random.sample((128, 16000))

u = np.dot(a, a.T) + 1e-12

w, v = eigh(u)
w = np.diag(w)
assert (np.dot(v, v.T) - np.identity(128) < 1e-10).all()
assert (np.abs(np.dot(np.dot(v, w), v.T) - u) < 1e-10).all()