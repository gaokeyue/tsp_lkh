from typing import Sequence
import numpy as np
from tsp_lkh.tsp_lkh_zy.quicksort_zy import quicksort


class TourArray:
    def __init__(self, route: list):
        self.size = len(route)
        if sorted(route) != list(range(self.size)):
            raise ValueError(f"Input must be a permutation of {self.size}")
        self.route = route.copy()
        self.inverse_route = [0, ] * self.size
        for i, k in enumerate(route):
            self.inverse_route[k] = i

    def neighbours(self, i):
        index = self.inverse_route[i]
        return self.route[(index+1) % self.size], self.route[index-1]

    def iter_links(self, include_reverse=True):
        """return all links in tour"""
        curr = 0
        while next != self.size:
            yield self.route[curr], self.route[curr+1]
            if include_reverse:
                yield self.route[curr+1], self.route[curr]
            curr += 1
        # return [(self.route[curr], self.route[curr+1]) for curr in range(self.size-1)]

    def check_feasible(self, v: list):
        # troublesome to modify LKH. TUT
        # O(n) if p_2k_2 empty, O(k) if not
        # Gao lao shi Chao qiang!
        # p_2k_2 stands for the permutation of the first (2k-2) break_vs in the route
        # if not p_2k_2:  # O(n) in worst case
        #     # set v0 in front of v1 in the route
        #     if self.route.index(v[0]) > self.route.index(v[1]):
        #         self.route.reverse()
        #         for i, k in enumerate(self.route):
        #             self.inverse_route[k] = i
        # # get the permutation of last two break_vs
        # node1 = len(v) - 2 if self.route.index(v[-1]) > self.route.index(v[-2]) else len(v) - 1
        # node2 = len(v) - 1 if node1 == len(v) - 2 else len(v) - 2
        # p_2k = p_2k_2.copy()
        # for i in p_2k_2:  # O(k)
        #     if self.route.index(v[i]) > self.route.index(v[node1]):
        #         p_2k.insert(i - 1, node1)
        #         p_2k.insert(i, node2)
        #         break
        # if len(p_2k) == len(p_2k_2):
        #     p_2k.append(node1)
        #     p_2k.append(node2)
        id0 = self.inverse_route[v[0]]
        id1 = self.inverse_route[v[1]]
        tmp = {}
        reversed_orientation = False if id0 < id1 else True
        for i in range(len(v)):
            tmp[self.inverse_route[v[i]]] = i
        keys = list(tmp.keys())
        quicksort(keys, reverse=reversed_orientation)
        idk = keys.index(self.inverse_route[v[0]])
        p_2k = []
        for i in keys[idk:]+keys[:idk]:
            p_2k.append(tmp[i])

        # q: the inverse permutation w.r.t. p_2k
        q = [0, ] * len(p_2k)
        for i, k in enumerate(p_2k):  # O(k)
            q[k] = i

        incl = [0, len(v) - 1]
        # First jump v[0] to v[2k-1] = v[-1]
        while len(incl) < len(v):  # O(k)
            index = q[incl[-1]]
            item2append = p_2k[index - 1] if index % 2 == 0 else p_2k[(index + 1) % len(v)]
            if item2append == 0:
                break
            incl.append(item2append)
            item2append = incl[-1] - 1 if incl[-1] % 2 == 0 else incl[-1] + 1
            if item2append == 0:
                break
            incl.append(item2append)
        return len(incl) == len(v)

    def k_exchange(self, v: list):  # O(n*k^2) ???
        id0 = self.inverse_route[v[0]]
        id1 = self.inverse_route[v[1]]
        if id0 == id1 + 1 or id1 == id0 + 1:
            initial_index = max(id0, id1)
            tmp_route = self.route[initial_index:] + self.route[:initial_index]  # O(n)
        else:
            tmp_route = self.route.copy()
        # cut the tour to get several fragment w.r.t. v
        fragments = []
        first_v_index = 0
        while first_v_index < self.size:  # O(n)
            next_v_index = first_v_index + 1
            while next_v_index < self.size:
                if tmp_route[next_v_index] not in v:
                    next_v_index += 1
                else:
                    next_v_index += 1
                    break
            fragments.append(tmp_route[first_v_index:next_v_index])
            first_v_index = next_v_index
        # relink the fragments into a new route
        pair2add = {}
        for i in range(len(v) // 2):  # O(k)
            pair2add[v[2 * i - 1]] = v[2 * i]
            pair2add[v[2 * i]] = v[2 * i - 1]
        new_route = []
        first_node = v[-1]
        for _ in range(len(fragments)):  # O(n*k^2)
            last_node = -1
            for fragment in fragments:  # O(n*k)
                if first_node == fragment[0]:
                    new_route.extend(fragment)
                    # fragments.remove(fragment)
                    last_node = fragment[-1]
                    break
                elif first_node == fragment[-1]:  # O(n) in worst case
                    new_route.extend(reversed(fragment))
                    # fragments.remove(fragment)
                    last_node = fragment[0]
                    break
            first_node = pair2add[last_node]
        self.route = new_route.copy()
        for i, k in enumerate(new_route):
            self.inverse_route[k] = i
        return

    def route_cost(self, cost: np.array):
        """return the cost of this tour"""
        curr = -1
        next = 0
        a = 0
        while next != self.size:
            a += cost[self.route[curr], self.route[next]]
            curr += 1
            next += 1
        return a


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
