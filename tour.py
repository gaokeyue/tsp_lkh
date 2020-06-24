from typing import Sequence
import numpy as np


def is_permutation_n(a_seq: Sequence, p0: Sequence = None):
    """Check whether a seq is a permutaion of range(n)."""
    if p0 is None:
        n = len(a_seq)
        attendence = np.zeros(n, bool)
        for x in a_seq:
            try:
                dao = attendence[x]
            except IndexError:
                print(f"{x} not in range({n})")
                return False
            else:
                if dao:
                    print(f"{x} is repeated")
                    return False
                attendence[x] = True
        assert attendence.all(), "Do I need this line?"
        return True
    attendence = {x: False for x in p0}
    for x in a_seq:
        dao = attendence[x]
        if dao:
            return False
        attendence[x] = True

class TourDoubleList:
    """Circular double linked list implementation of tour."""

    def __init__(self, route: Sequence):
        """Accept any sequence of permutation of range(n). """
        # self.route = route
        self.size = len(route)
        if route == [0] * self.size or route == [-1] * self.size:
            # Permit null initialization.
            self.links = np.zeros((self.size, 2), int)
        elif sorted(route) != list(range(self.size)):
            raise ValueError(f"Input must be a permutation of {self.size}")
        else:
            # convert the sequence to doubly linked list.
            # self.links[i, j] is the predecessor of i if j is 0, successor of i if j is 1
            self.links = np.zeros((self.size, 2), dtype=int)
            for i in range(self.size):
                self.links[route[i]] = route[i - 1], route[i]

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

    def routine_cost(self, cost: np.array):
        """return the cost of this tour"""
        start = 0
        next = self.links[start, 1]
        a = cost[start, next]
        curr = next
        while curr != start:
            next = self.links[curr, 1]
            a += cost[curr, next]
            curr = next
        return a
