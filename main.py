import os
import numpy as np

DATA_DIR = 'data/'


def tour_cost_arr(tour, cost_mat):
    """Compute the cost a tour of array representation"""
    total_cost = 0
    for i in range(len(tour)):
        total_cost += cost_mat[tour[i - 1], tour[i]]
    return total_cost


def read_test_data(cost_file, answer_file):
    cost_mat = np.load(DATA_DIR + cost_file)
    n = len(cost_mat)
    cost_mat[range(n), range(n)] = np.inf
    opt_tour = np.load(DATA_DIR + answer_file)
    return cost_mat, opt_tour


class LK:
    pass


class LKH:
    pass


if __name__ == '__main__':
    cost_mat, opt_tour = read_test_data('ch130.npy', 'ch_ans.npy')
    print(f"The optimal cost should be {tour_cost_arr(opt_tour, cost_mat)}")
