from typing import Sequence
import numpy as np


class TourDoubleList:
    """Doubly circular linked list implementation of tour."""

    def __init__(self, route: Sequence):
        """Accept any sequence of permutation of range(n). """
        # self.route = route
        self.size = len(route)
        route = list(route)
        if route == [0] * self.size or route == [-1] * self.size:
            # Permit null initialization.
            self.links = np.zeros((self.size, 2), int)
        elif sorted(route) != list(range(self.size)):
            raise ValueError(f"Input must be a permutation of {self.size}")
        else:
            # convert the sequence to doubly linked list.
            # self.links[i, j] is the predecessor of i if j is 0, successor of i if j is 1
            self.links = np.zeros((self.size, 2), int)
            last = route[0]
            for curr in route[1:]:
                self.links[last, 1] = curr
                self.links[curr, 0] = last
                last = curr
            self.links[last, 1] = route[0]
            self.links[route[0], 0] = last

    # def clone(self):
    #     """Copy the tour to a new one"""
    #     return TourDoubleList(self.route)

    def next(self, i):
        return self.links[i, 1]

    def prev(self, i):
        return self.links[i, 0]

    def neighbours(self, i):
        return self.links[i]

    def iter_vertices(self, start=0, reverse=False):
        """return the route as a sequence"""
        yield start
        orientation = 0 if reverse else 1
        successor = self.links[start, orientation]
        while successor != start:
            yield successor
            successor = self.links[successor, orientation]

    def iter_links(self, include_reverse=True):
        """return all links in tour"""
        start = 0
        curr = start
        next = self.links[curr, 1]
        while next != start:
            yield curr, next
            if include_reverse:
                yield next, curr
            curr, next = next, self.links[next, 1]

    def check_feasible(self, v):
        """Walk around the tour, O(n) complexity"""
        """
        v: sequential series, [v_0, v_1, ..., V_2k-2, v_2k-1]
        """
        l = len(v)
        if len(set(v)) != l:
            raise ValueError("Repeat elements in sequential series!")
        if l < 4 or l % 2 != 0:
            raise ValueError("The number of sequential series should be an odd, at least 4!")
        cnt = 0
        start = 2
        curr = 2
        direction = int(self.links[v[curr], 1] == v[curr + 1])
        while cnt <= self.size + 1:
            curr = self.links[v[curr], 1 - direction]  # curr is a position
            cnt += 1
            while curr not in v:
                curr = self.links[curr, 1 - direction]  # curr is a position
                cnt += 1
            if v.index(curr) % 2 == 1:
                curr = (v.index(curr) + 1) % l  # curr is an index
                direction = int(self.links[v[curr], 1] == v[curr + 1])  # now curr is odd
            else:
                curr = (v.index(curr) - 1) % l  # curr is an index
                direction = int(self.links[v[curr], 1] == v[curr - 1])  # now curr is odd
            cnt += 1
            if curr == start:
                break
        if cnt == self.size:
            return True
        else:
            return False

    def k_exchange(self, v: list):
        """
        Make a k-exchange w.r.t. sequential mv = [v_0, v_1, ..., v_2k-2, v_2k-1],
        where we add (v_2i-1, v_2i) and remove (v_2i, v_2i+1).
        ::type mv:: length-4 list.
        ::rtype:: Return a new Tour object, or None."""
        l = len(v)
        if len(set(v)) != l:
            raise ValueError("Repeat elements in sequential series!")
        if l < 4 or l % 2 != 0:
            raise ValueError("The number of sequential series should be an odd, at least 4.")
        newone = TourDoubleList([0 for i in range(self.size)])
        cnt = 0
        start = 2
        curr = 2
        direction = int(self.links[v[curr], 1] == v[curr + 1])
        while True:
            prev = v[curr]
            curr = self.links[v[curr], 1 - direction]  # curr is a position
            newone.links[prev, 1], newone.links[curr, 0] = curr, prev
            cnt += 1
            while curr not in v:
                prev = curr
                curr = self.links[curr, 1 - direction]  # curr is a position
                newone.links[prev, 1], newone.links[curr, 0] = curr, prev
                cnt += 1
            prev = curr
            if v.index(curr) % 2 == 1:
                curr = (v.index(curr) + 1) % l  # curr is an index
                direction = int(self.links[v[curr], 1] == v[curr + 1])  # now curr is odd
            else:
                curr = (v.index(curr) - 1) % l  # curr is an index
                direction = int(self.links[v[curr], 1] == v[curr - 1])  # now curr is odd
            newone.links[prev, 1], newone.links[v[curr], 0] = v[curr], prev
            cnt += 1
            if curr == start:
                break
        return newone

    def two_optimal(self, mv: list):
        """
        Make a two-exchange w.r.t. edge (v_0, v_1) = (mv[0], mv[1]),  (v_2, v_3) = (mv[2], mv[3]) (to be removed)
        and edge (v_1, v_2), (v_3, v_0) (to be added).
        ::type mv:: length-4 list.
        ::rtype:: Return a new Tour object, or None.
        """
        v0 = mv[0]
        v1 = mv[1]
        if self.next(v0) == v1:
            direction = 1
        elif self.prev(v0) == v1:
            direction = 0
        else:
            raise ValueError('Error: Only move the edge between neighbor nodes.')
        v2 = mv[2]
        v3 = mv[3]
        if v3 != self.links[v2, 1 - direction]:
            raise ValueError('Error: Unfeasible position of the second vertice to be broken.')
        # Get rid of bad cases.
        # if v2 == -1 and v3 == -1:
        #     return
        if v2 == v0 or v2 == v1 or v3 == v0 or v3 == v1:
            return
        newone = TourDoubleList([0 for i in range(self.size)])
        # print("Clone one:", list(newone.iter_vertices()))
        curr = v3
        # TicToc.tic()
        """Found by Gao. Exchange prev. and next of ref. vertices."""
        while curr != v0:
            newone.links[curr, 0], newone.links[curr, 1] = self.links[curr, 1], self.links[curr, 0]
            curr = self.links[curr, 1 - direction]
        while curr != v3:
            newone.links[curr] = self.links[curr]
            curr = self.links[curr, 1 - direction]
        # TicToc.toc()
        newone.links[v1, direction] = v2
        newone.links[v2, 1 - direction] = v1
        newone.links[v0, direction] = v3
        newone.links[v3, 1 - direction] = v0
        # print("Original one:", list(self.iter_vertices()))
        # print("Changed Clone one:", list(newone.iter_vertices()))
        return newone

    def route_cost(self, cost: np.array):
        """return the cost of this tour"""
        start = 0
        nnext = self.links[start, 1]
        a = cost[start, nnext]
        curr = nnext
        while curr != start:
            nnext = self.links[curr, 1]
            a += cost[curr, nnext]
            curr = nnext
        return a
