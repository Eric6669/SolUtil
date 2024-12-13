import numpy as np


def calrmse(a, b):
    nz = a.shape[0]
    return np.sqrt(np.sum((a - b) ** 2, axis=0) / nz)

