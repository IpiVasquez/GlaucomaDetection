import numpy as np

def cdr(disc, cup):
    return np.count_nonzero(cup != 0) / np.count_nonzero(disc != 0)
