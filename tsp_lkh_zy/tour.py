from typing import Sequence
import numpy as np


class TourDoubleList:
    """Doubly circular linked list implementation of tour."""

    def __init__(self, route: Sequence):
        """Accept any sequence of permutation of range(n). """
        self.size = len(route)
        if sorted(route) != list(range(self.size)):
            raise ValueError(f"Input must be a permutation of {self.size}")
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

    def succ(self, i):
        return self.links[i, 1]

    def pred(self, i):
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
        successor = self.links[curr, 1]
        while successor != start:
            yield curr, successor
            if include_reverse:
                yield successor, curr
            curr, successor = successor, self.links[successor, 1]

    def check_feasible(self, v):
        """Walk around the tour, O(n) complexity"""
        # step 1, determine the order of v[0], v[1],...,v[2k-1] with orientation v[0]-> v[1] in self.
        p = [0]  # Fix p[0] = 0, and p[1] = 1
        orientation = 1 if self.links[v[0], 1] == v[1] else 0
        start = v[0]
        successor = self.links[start, orientation]
        while successor != v[0]:
            if successor in v:
                for i, k in enumerate(v):
                    if successor == k:
                        p.append(i)
            successor = self.links[successor, orientation]

        q = [0 for _ in range(len(v))]  # the inverse permutation of v, q = [p.index(i) for i in range(2k)]
        for i, k in enumerate(p):
            q[k] = i

        incl = [0, len(v)-1]
        # First jump v[0] to v[2k-1] = v[-1]
        while len(incl) < len(v):
            index = q[incl[-1]]
            item2append = p[index-1] if index % 2 == 0 else p[(index+1) % len(v)]
            if item2append == 0:
                break
            incl.append(item2append)
            item2append = incl[-1]-1 if incl[-1] % 2 == 0 else incl[-1]+1
            if item2append == 0:
                break
            incl.append(item2append)
        return len(incl) == len(v)

    def k_exchange(self, v):
        # walk around the tour, check and change the orientation when meet break_vs
        new_links = np.zeros_like(self.links, dtype=int)
        pair2del = {}
        pair2add = {}
        for i in range(len(v)//2):
            pair2del[v[2*i]] = v[2*i+1]
            pair2del[v[2*i+1]] = v[2*i]
            pair2add[v[2*i-1]] = v[2*i]
            pair2add[v[2*i]] = v[2*i-1]
        new_links[v[0], 1] = v[-1]
        new_links[v[-1], 0] = v[0]
        curr_v = v[-1]
        orientation = 0 if self.links[curr_v, 1] == pair2del[curr_v] else 1
        next_v = self.links[curr_v, orientation]
        while True:
            new_links[curr_v, 1] = next_v
            new_links[next_v, 0] = curr_v
            if next_v not in pair2del:  # you have to crawl
                curr_v = next_v
                next_v = self.links[curr_v, orientation]
            else:  # leap
                curr_v = next_v
                next_v = pair2add[curr_v]
                new_links[curr_v, 1] = next_v
                new_links[next_v, 0] = curr_v
                if next_v == v[-1]:
                    break
                curr_v = next_v
                orientation = 0 if self.links[curr_v, 1] == pair2del[curr_v] else 1
                next_v = self.links[curr_v, orientation]
            # else:  # you have to crawl, sorry
        self.links = new_links
        return

        # self.links[v[0], orientation], self.links[v[-1], 1-orientation] = v[-1], v[0]
        # curr_v = self.links[v[0], orientation]
        # while curr_v != v[0]:
        #     if curr_v in v:
        #         index = -1
        #         for i, k in enumerate(v):
        #             if curr_v == k:
        #                 index = i
        #         if index % 2 == 0:
        #             orientation = 1 if self.links[curr_v, 1] == v[index+1] else 0
        #         else:
        #             orientation = 1 if self.links[curr_v, 1] == v[index-1] else 0
        #     node = self.links[curr_v, orientation]
        #     self.links[curr_v, orientation], self.links[node, 1-orientation] = node, curr_v
        #     curr_v = node
        # return
