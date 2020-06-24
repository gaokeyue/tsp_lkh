from __future__ import annotations
import numpy as np
from heapq import nsmallest
from operator import itemgetter
# from collections import defaultdict
import random
cd .
from classic_lkh_zy.tsp_lkh.tour import TourDoubleList as Tour
from classic_lkh_zy.tsp_lkh.search_for_d import alpha_nearness
from classic_lkh_zy.tsp_lkh.prim import CompleteGraph, PrimVertex


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

    def alpha(self):
        graph = CompleteGraph(self.cost)
        return alpha_nearness(graph)

    def creat_initial_tour(self):
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

    def tour_cost(self, tour: Tour):
        length = 0
        v = 0
        while tour.links[v, 1] != 0:
            length += self.cost[v, tour.links[v, 0]]
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
        i = int(len(sque_v)/2)  # step i done already
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
                        return self.dfs_recursion(tour, sque_v + [v_2i, v_2ip1], new_gain)
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


def create_test_cost(n: int):
    np.random.seed(3)
    a = np.random.rand(n, n)
    a = np.tril(a).T + np.tril(a)
    a[range(n), range(n)] = np.inf
    return a


n = 100
my_cost = create_test_cost(n)

test_lkh = TSP_LKH(my_cost, submove_size=5)
original_tour = Tour(list(range(n)))
print("Original tour: ", list(original_tour.iter_vertices()))
print("Original cost: ", test_lkh.tour_cost(original_tour))
test_tour = test_lkh.creat_initial_tour()
print("Initial tour: ", list(test_tour.iter_vertices()))
print("Initial cost: ", test_lkh.tour_cost(test_tour))
# best = test_lkh.local_optimum(original_tour)
best = test_lkh.local_optimum(test_tour)
