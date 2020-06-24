from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Dict, Tuple, Set
from itertools import permutations
import numpy as np
import random
from .tour import TourDoubleList as Tour
DATA_DIR = 'data'


class LK:
    def __init__(self, cost: np.array):
        self.cost = cost  # symmetric n-by-n numpy array
        self.size = len(cost)
        self.max_exchange = 5
        self.nearest_num = 5
        # candidates store the five nearest vertices of each nodes.
        self.candidates = {}  # self.candidates must support __getitem__

    def create_test_candidates(self):
        for i in range(self.size):
            self.candidates[i] = [j for j in range(self.size) if j != i]

    def run(self, tour: Tour):
        """improve an initial tour by variable-exchange until local optimum."""
        better = tour
        cnt = 0
        while better is not None:
            print("---------------------------------------------\n The ", cnt, "time.")
            print("Improved tour:", list(tour.iter_vertices()))
            print("Improved cost:", tour.routine_cost(self.cost))
            cnt += 1
            tour = self.improve(tour)
            better = tour
        return tour

    def improve(self, tour: Tour):
        """
        The head function to use iterable func. dfs_iter
        """
        # i, j = 0, 0
        # better = self.dfs_iter(tour, [i, tour.links[i, j]], self.cost[i, tour.links[i, j]])
        better = None
        for i in range(tour.size):
            for j in [0, 1]:
                if better is not None:
                    return better
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
        self.maxg = 0
        self.bestseq = None

    def create_test_candidates(self):
        for i in range(self.size):
            self.candidates[i] = [j for j in range(self.size) if j != i]

    def run(self, tour: Tour):
        """improve an initial tour by variable-exchange until local optimum."""
        cnt = 0
        better = None
        while tour is not None:
            better = tour
            print("---------------------------------------------\n The ", cnt, "time.")
            print("Improved tour:", list(better.iter_vertices()))
            print("Improved cost:", better.routine_cost(self.cost))
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
        mycand = self.candidates[v_2i_1]
        for v_2i in mycand:
            if v_2i not in seqv and gain - self.cost[v_2i_1, v_2i] > 0:
                # Choose v_{2i+1} from neighbours of v_{2i-1}.
                for v_2i__1 in tour.links[v_2i]:
                    if v_2i__1 not in seqv:
                        # Calculate potential gain w.r.t. edge (v_{2i+1}, v_0)
                        del_gain = - self.cost[v_2i_1, v_2i] + self.cost[v_2i, v_2i__1]
                        if k % self.max_exchange == 0:
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
                                return self.dfs_iter(tour, seqv + [v_2i, v_2i__1], gain + del_gain)
        return


if __name__ == '__main__':
    cost_filepath = 'data/ch130.npy'
    cost_mat = np.load(cost_filepath)
    n = len(cost_mat)
    cost_mat[range(n), range(n)] = np.inf