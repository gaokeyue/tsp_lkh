import os
import numpy as np
DATA_DIR = 'data'


class LK:
    pass

class LKH:
    pass

if __name__ == '__main__':
    cost_filepath = 'data/ch130.npy'
    cost_mat = np.load(cost_filepath)
    n = len(cost_mat)
    cost_mat[range(n), range(n)] = np.inf