from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Dict, Tuple, Set
from itertools import permutations
import numpy as np
import random
from tsp_lkh_zy.tour import TourDoubleList as Tour
from heapq import nsmallest
from operator import itemgetter
from tsp_lkh_zy.prim import CompleteGraph
from tsp_lkh_zy.search_for_d import alpha_nearness, best_pi
DATA_DIR = 'data'


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


def preprocess_cost_mat(cost_mat):
    """Zhao Ying nu li ban zhuan"""
    graph = CompleteGraph(cost_mat)
    pi = best_pi(graph)
    cost_mat_d = cost_mat.copy()
    for i in range(len(pi)):
        for j in range(i, len(pi)):
            cost_mat_d[i, j] = cost_mat_d[j, i] = cost_mat[i, j] + pi[i] + pi[j]
    alpha_value = alpha_nearness(graph)
    return cost_mat_d, alpha_value, pi


class LK:
    def __init__(self, cost: np.array):
        self.cost = cost  # symmetric n-by-n numpy array
        self.size = len(cost)
        self.max_exchange = 5
        self.nearest_num = 5
        # candidates store the five nearest vertices of each nodes.
        self.candidates = {}  # self.candidates must support __getitem__
        self.create_test_candidates()

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


class TSP_LKH:
    def __init__(self, cost: np.array, submove_size=5):
        self.cost = cost  # symmetric n-by-n numpy array
        self.size = len(cost)
        self.max_exchange = submove_size
        self.nearest_num = 5
        self.candidates = {}  # self.candidates must support __getitem__
        self.creat_candidates()

    def creat_candidates(self):
        self.candidates = {}
        for i in range(self.size):
            sorted_nodes = nsmallest(self.nearest_num+1, enumerate(self.cost[i]), key=itemgetter(1))
            tmp_lst = [v for v, _ in sorted_nodes if v != i]
            if len(tmp_lst) > self.nearest_num:
                tmp_lst.pop()
            self.candidates[i] = tmp_lst

    def creat_initial_tour(self, alpha_value):
        tour = Tour(list(range(self.size)))
        i = random.randint(0, self.size-1)
        new_links = np.zeros((self.size, 2), int)
        done = [i]
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

    def tour_cost(self, tour: Tour):
        v = 0
        length = self.cost[v, tour.links[v, 0]]
        while tour.links[v, 1] != 0:
            length += self.cost[v, tour.links[v, 1]]
            v = tour.links[v, 1]
        return length

    def improve(self, tour: Tour):
        """Improve a tour by a variable-exchange, at most 5-exchange.
        We would like the three variables, i, break_vs and gain, to be shared by all recursion calls,
        so two options are available to deal with the problem that i and gain are immutable:
        (1) place the recursion function inside the improve function (as an inner function) and use the nonlocal
        trick. The nonlocal label is effective throught all recursions;
        (2) place the recursion function outside the improve function, but wrap all the variables inside
        a mutable variable, e.g a dictionary, and then pass this dummy mutable variable into all recursions.
        Approach (2) is the one adopted here, because it's probably more flexible,
        in case the recursion function will be called by another function in addition to the improve function"""
        for v0, v1 in tour.iter_links():
            result = self.dfs_recursion(tour, [v0, v1], 0)
            if result is not None:
                return result

    def dfs_recursion(self, tour, sque_v, gain):
        """depth-first-search by recursion called by self.improve.
        If a feasible and profitable tour is found beyond break_vs = [v1, v2,..., v_(2i-1), v_(2i)],
        this function returns the tour. Otherwise return None."""
        i = len(sque_v) // 2  # step i done already

        if i == 10:
            return

        dahuitou = (i + 1) % self.max_exchange == 0
        v_2i_2, v_2i_1 = sque_v[-2], sque_v[-1]
        # step i+1: search for (v_2i, v_2ip1)
        for v_2i in self.candidates[v_2i_1]:
            if v_2i in sque_v:  # disjunctivity criterion
                continue
            new_gain = gain + self.cost[v_2i_2, v_2i_1] - self.cost[v_2i_1, v_2i]
            if new_gain <= 0:
                continue
            for v_2ip1 in tour.neighbours(v_2i):
                if v_2ip1 in sque_v:  # disjunctivity criterion
                    continue
                if dahuitou:
                    if tour.check_feasible(sque_v+[v_2i, v_2ip1]):
                        if new_gain + self.cost[v_2i, v_2ip1] - self.cost[v_2ip1, sque_v[0]] > 0:
                            tour.k_exchange(sque_v + [v_2i, v_2ip1])
                            return tour
                        else:
                            return self.dfs_recursion(tour, sque_v + [v_2i, v_2ip1], new_gain)
                    else:  # optional, can be deleted
                        continue
                else:
                    if new_gain + self.cost[v_2i, v_2ip1] - self.cost[v_2ip1, sque_v[0]] > 0 and \
                            tour.check_feasible(sque_v+[v_2i, v_2ip1]):
                        tour.k_exchange(sque_v + [v_2i, v_2ip1])
                        return tour
                    else:
                        result = self.dfs_recursion(tour, sque_v + [v_2i, v_2ip1], new_gain)
                        if result is not None:
                            return result
                # if (i + 1) % self.max_exchange == 0:
                #     continue
                # return self.dfs_recursion(tour, sque_v + [v_2i, v_2ip1], new_gain)
                # gain += - self.cost[v_2i_1, v_2i] + self.cost[v_2i, v_2ip1]
                # if gain - self.cost[v_2ip1, sque_v[0]] > 0:
                #     # check feasibility immediately
                #     if tour.check_feasible(sque_v+[v_2i, v_2ip1]):
                #         tour.k_exchange(sque_v + [v_2i, v_2ip1])
                #         return tour
                #     # if not feasible, check whether stop or search for next two nodes
                #     if (i+1) % self.max_exchange == 0:
                #         continue
                #     delta_gain = self.cost[sque_v[2 * i - 2], sque_v[2 * i - 1]] - self.cost[sque_v[2 * i - 1], v_2i]
                #     return self.dfs_recursion(tour, sque_v+[v_2i, v_2ip1], gain+delta_gain)
        return

    def local_optimum(self, tour: Tour):
        """improve an initial tour by variable-exchange until local optimum."""
        better = tour
        cnt = 0
        while better is not None:
            # if cnt % 20 == 0:
            print("---------------------------------------------\n The ", cnt, "time.")
            print("Improved tour:", list(better.iter_vertices()))
            print("Improved cost:", self.tour_cost(better))
            cnt += 1
            tour = better
            better = self.improve(tour)
        return tour


if __name__ == '__main__':
    cost_filepath = 'data/ch130.npy'
    cost_mat = np.load(cost_filepath)
    n = len(cost_mat)
    cost_mat[range(n), range(n)] = np.inf
    cost_mat_d, alpha_value, pi = preprocess_cost_mat(cost_mat)
    lkh_file = TSP_LKH(cost_mat_d)
    lk_file = LK(cost_mat_d)
    original_tour = Tour(list(range(n)))
    print("Original tour: ", list(original_tour.iter_vertices()))
    print("Original cost: ", lkh_file.tour_cost(original_tour))

    test_tour = lkh_file.creat_initial_tour(alpha_value)
    print("Initial tour: ", list(test_tour.iter_vertices()))
    print("Initial cost: ", lkh_file.tour_cost(test_tour))
    # best = test_lkh.local_optimum(original_tour)
    best = lkh_file.local_optimum(test_tour)
