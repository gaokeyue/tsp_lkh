# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Dict, Tuple, Set
from itertools import permutations
import numpy as np
import random
from tour import TourDoubleList as Tour
import time
from tsp_lkh_zy.search_for_d import alpha_nearness
from tsp_lkh_zy.prim import CompleteGraph, PrimVertex
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


# def preprocess_cost_mat(cost_mat):
#     """ZHao ying da xian shen shou..."""
#     return cost_mat_beta, alpha_near


class LK:
    def __init__(self, cost: np.array):
        self.cost = cost  # symmetric n-by-n numpy array
        self.size = len(cost)
        self.max_exchange = 5
        self.nearest_num = 5
        # candidates store the five nearest vertices of each nodes.
        self.candidates = {}  # self.candidates must support __getitem__
        self.create_candidates()

    def create_test_candidates(self):
        for i in range(self.size):
            self.candidates[i] = [j for j in range(self.size) if j != i]

    def make_knn_candidates(self, k=5):
        for i in range(self.size):
            self.candidates[i] = np.argsort(self.cost[i])[:k]

    def create_candidates(self):
        self.candidates = {}
        for i in range(self.size):
            sorted_nodes = nsmallest(self.nearest_num+1, enumerate(self.cost[i]), key=itemgetter(1))
            tmp_lst = [v for v, _ in sorted_nodes if v != i]
            if len(tmp_lst) > self.nearest_num:
                tmp_lst.pop()
            self.candidates[i] = tmp_lst

    def alpha(self):
        graph = CompleteGraph(self.cost)
        return alpha_nearness(graph)

    def create_initial_tour(self):
        tour = Tour(list(range(self.size)))
        i = random.randint(0, self.size-1)
        new_links = np.zeros((self.size, 2), int)
        done = [i]
        alpha_value = self.alpha()
        while len(done) < self.size:
            new_i = -1
            for j in range(self.size):
                if j in done:
                    continue
                if j in self.candidates[i] and alpha_value[i, j] == 0:
                    new_links[i, 1] = j
                    new_links[j, 0] = i
                    new_i = j
                    done.append(j)
                    break
            if new_i == -1:
                for j in range(self.size):
                    if j in done:
                        continue
                    if j in self.candidates[i]:
                        new_links[i, 1] = j
                        new_links[j, 0] = i
                        new_i = j
                        done.append(j)
                        break
            if new_i == -1:
                for j in range(self.size):
                    if j not in done:
                        new_links[i, 1] = j
                        new_links[j, 0] = i
                        new_i = j
                        done.append(j)
                        break
            i = new_i
        last = done[-1]
        new_links[last, 1] = done[0]
        new_links[done[0], 0] = last
        tour.links = new_links
        return tour

    def run(self):
        """improve an initial tour by variable-exchange until local optimum."""
        tour = self.create_initial_tour()
        better = None
        cnt = 0
        while tour is not None:
            # print("---------------------------------------------\n The ", cnt, "time.")
            # print("Improved tour:", list(tour.iter_vertices()))
            # print("Improved cost:", tour.routine_cost(self.cost))
            cnt += 1
            better = tour
            tour = self.improve(better)
        return better

    def improve(self, tour: Tour):
        """
        The head function to use iterable func. dfs_iter
        """
        # i, j = 0, 0
        # better = self.dfs_iter(tour, [i, tour.links[i, j]], self.cost[i, tour.links[i, j]])
        for i in range(tour.size):
            for j in [0, 1]:
                better = self.dfs_iter(tour, [i, tour.links[i, j]], self.cost[i, tour.links[i, j]])
                if better is not None:
                    return better
        return

    def dfs_iter(self, tour: Tour, seqv: list, gain: int):
        """
        tour: Updated tour w.r.t. SeqV
        seqv: sequential vertices (v_0, v_1, ..., v_{2i-2}, v_{2i-1})
        gain: partial sum to (i-1) of |x| - |y| plus |x_i|
        """
        v_0 = seqv[0]
        v_2i_1 = seqv[-1]
        # gain += self.cost[seqv[-1], seqv[-2]]
        mycand = self.candidates[v_2i_1]
        for v_2i in mycand:
            if v_2i not in seqv and gain - self.cost[v_2i_1, v_2i] > 0:
                # Decide v_{2i+1} according to (v_0, v_{2i-1}).
                if tour.next(v_0) == v_2i_1:
                    direction = 1
                elif tour.prev(v_0) == v_2i_1:
                    direction = 0
                else:
                    raise ValueError('Error: Not feasible (v_0, v_{2i-1}) in given tour.')
                v_2i__1 = tour.links[v_2i, 1 - direction]
                if v_2i__1 not in seqv:
                    # Calculate potential gain w.r.t. edge (v_{2i+1}, v_0)
                    del_gain = - self.cost[v_2i_1, v_2i] + self.cost[v_2i, v_2i__1]
                    if gain + del_gain - self.cost[v_2i__1, v_0] > 0:
                        return tour.two_optimal([v_0, v_2i_1, v_2i, v_2i__1])
                    return self.dfs_iter(tour.two_optimal([v_0, v_2i_1, v_2i, v_2i__1]), seqv + [v_2i, v_2i__1], gain + del_gain)
        return


class LKH:
    def __init__(self, cost: np.array):
        self.cost = cost  # symmetric n-by-n numpy array
        self.size = len(cost)
        self.max_exchange = 5
        self.nearest_num = 5
        # candidates store the five nearest vertices of each nodes.
        self.candidates = {}  # self.candidates must support __getitem__
        self.make_knn_candidates()

    def create_test_candidates(self):
        for i in range(self.size):
            self.candidates[i] = [j for j in range(self.size) if j != i]

    def make_knn_candidates(self, k=5):
        for i in range(self.size):
            self.candidates[i] = np.argsort(self.cost[i])[:k]

    def run(self, tour: Tour):
        """improve an initial tour by variable-exchange until local optimum."""
        cnt = 0
        better = None
        while tour is not None:
            better = tour
            # print("---------------------------------------------\n The ", cnt, "time.")
            # print("Improved tour:", list(better.iter_vertices()))
            # print("Improved cost:", better.routine_cost(self.cost))
            cnt += 1
            tour = self.improve(better)
        return better

    def improve(self, tour: Tour):
        """
        The head function to use iterable func. dfs_iter
        """
        # i, j = 0, 0
        # better = self.dfs_iter(tour, [i, tour.links[i, j]], self.cost[i, tour.links[i, j]])
        for i in range(tour.size):
            for j in [0, 1]:
                better = self.dfs_iter(tour, [i, tour.links[i, j]], self.cost[i, tour.links[i, j]])
                if better is not None:
                    return better
        return

    def dfs_iter(self, tour: Tour, seqv: list, gain: int):
        """
        tour: Original tour without updation.
        seqv: sequential vertices (v_0, v_1, ..., v_{2i-2}, v_{2i-1})
        gain: partial sum to (i-1) of |x| - |y| plus |x_i|
        """
        v_0 = seqv[0]
        v_2i_1 = seqv[-1]
        k = len(seqv) // 2
        if k == 10:
            return
        mycand = self.candidates[v_2i_1]
        for v_2i in mycand:
            if v_2i not in seqv and gain - self.cost[v_2i_1, v_2i] > 0:
                # Choose v_{2i+1} from neighbours of v_{2i-1}.
                for v_2i__1 in tour.links[v_2i]:
                    if v_2i__1 not in seqv:
                        # Calculate potential gain w.r.t. edge (v_{2i+1}, v_0)
                        del_gain = - self.cost[v_2i_1, v_2i] + self.cost[v_2i, v_2i__1]
                        if k+1 % self.max_exchange == 0:
                            if tour.check_feasible(seqv + [v_2i, v_2i__1]):
                                if gain + del_gain - self.cost[v_2i__1, v_0] > 0:
                                    return tour.k_optimal(seqv + [v_2i, v_2i__1])
                                else:
                                    return self.dfs_iter(tour, seqv + [v_2i, v_2i__1], gain + del_gain)
                        else:
                            if gain + del_gain - self.cost[v_2i__1, v_0] > 0:
                                if tour.check_feasible(seqv + [v_2i, v_2i__1]):
                                    return tour.k_optimal(seqv + [v_2i, v_2i__1])
                            else:
                                result = self.dfs_iter(tour, seqv + [v_2i, v_2i__1], gain + del_gain)
                                if result is not None:
                                    return result
        return


class TicToc(object):

    __tic_time, __toc_time = None, None

    @classmethod
    def tic(cls):
        cls.__tic_time = time.time()

    @classmethod
    def toc(cls):
        cls.__toc_time = time.time()
        print("\rTask Elapsed Time:   {0}\n".format(cls.__toc_time - cls.__tic_time))


"""_--------------------TEST-----------------------"""
if __name__ == '__main__':
    cost_mat, opt_tour = read_test_data('ch130.npy', 'ch_ans.npy')
    test_lk = LK(cost_mat)
    test_lkh = LKH(cost_mat)
    tour0 = list(range(len(cost_mat)))
    s = 34
    random.seed(s)
    random.shuffle(tour0)
    mytour = Tour(tour0)
    print(f"initial tour is {list(mytour.iter_vertices())}")
    TicToc.tic()
    tour_lk = test_lk.run(mytour)
    print("LK Improved tour:", list(tour_lk.iter_vertices()))
    print("LK Improved cost:", tour_lk.routine_cost(cost_mat))
    TicToc.toc()
    print(f"initial tour is {list(mytour.iter_vertices())}")
    # print("LK Improved cost:", tour_lk.routine_cost(cost_mat))
    # TicToc.tic()
    # tour_lkh = test_lkh.run(mytour)
    # print("LKH Improved tour:", list(tour_lkh.iter_vertices()))
    # print("LKH Improved cost:", tour_lkh.routine_cost(cost_mat))
    # TicToc.toc()
    # print(f"The optimal cost should be {tour_cost_arr(opt_tour, cost_mat)}")
