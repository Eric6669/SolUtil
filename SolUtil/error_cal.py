import numpy as np


def calrmse(a, b):
    a = np.array(a).reshape((-1, ))
    b = np.array(b).reshape((-1, ))
    nz = a.shape[0]
    return np.sqrt(np.sum((a - b) ** 2, axis=0) / nz)

