from typing import Sequence
import numpy as np


class TourArray:
    def __init__(self, route: list):
        self.size = len(route)
        if sorted(route) != list(range(self.size)):
            raise ValueError(f"Input must be a permutation of {self.size}")
        self.route = route.copy()
        self.inverse_route = [0, ]*self.size
        for i, k in enumerate(route):
            self.inverse_route[k] = i

    def check_feasible(self, v: list, p_2k_2: list):  # O(n) if p_2k_2 empty, O(k) if not
        # Gao lao shi Chao qiang!
        # p_2k_2 stands for the permutation of the first (2k-2) break_vs in the route
        if not p_2k_2:  # O(n) in worst case
            # set v0 in front of v1 in the route
            if self.route.index(v[0]) > self.route.index(v[1]):
                self.route.reverse()
                for i, k in enumerate(self.route):
                    self.inverse_route[k] = i
        # get the permutation of last two break_vs
        node1 = len(v)-2 if self.route.index(v[-1]) > self.route.index(v[-2]) else len(v)-1
        node2 = len(v)-1 if node1 == len(v)-2 else len(v)-2
        p_2k = p_2k_2.copy()
        for i in p_2k_2:  # O(k)
            if self.route.index(v[i]) > self.route.index(v[node1]):
                p_2k.insert(i-1, node1)
                p_2k.insert(i, node2)
                break
        if len(p_2k) == len(p_2k_2):
            p_2k.append(node1)
            p_2k.append(node2)
        # q: the inverse permutation w.r.t. p_2k
        q = [0, ]*len(p_2k)
        for i, k in enumerate(p_2k):  # O(k)
            q[k] = i

        incl = [0, len(v)-1]
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
        initial_index = max(self.route.index(v[0]), self.route.index(v[1]))
        tmp_route = self.route[initial_index:] + self.route[:initial_index]  # O(n)
        # cut the tour to get several fragment w.r.t. v
        fragments = []
        first_v_index = 0
        while first_v_index < self.size:  # O(n)
            next_v_index = first_v_index+1
            while next_v_index < self.size:
                if tmp_route[next_v_index] not in v:
                    next_v_index += 1
                else:
                    break
            fragments.append(tmp_route[first_v_index:next_v_index+1])
            first_v_index = next_v_index + 1
        # relink the fragments into a new route
        # pair2del = {}
        pair2add = {}
        for i in range(len(v) // 2):  # O(k)
            # pair2del[v[2 * i]] = v[2 * i + 1]
            # pair2del[v[2 * i + 1]] = v[2 * i]
            pair2add[v[2 * i - 1]] = v[2 * i]
            pair2add[v[2 * i]] = v[2 * i - 1]
        new_route = []
        first_node = v[-1]
        for _ in range(len(fragments)):  # O(n*k^2)
            last_node = -1
            for fragment in fragments:  # O(n*k)
                if first_node == fragment[0]:
                    new_route += fragment
                    fragments.remove(fragment)
                    last_node = fragment[-1]
                    break
                elif first_node == fragment[-1]:  # O(n) in worst case
                    new_route += reversed(fragment)
                    fragments.remove(fragment)
                    last_node = fragment[0]
                    break
            first_node = pair2add[last_node]
        self.route = new_route.copy()
        for i, k in enumerate(new_route):
            self.inverse_route[k] = i
        return


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
        start = v[0]
        p = [0]  # Fix p[0] = 0, and p[1] = 1
        orientation = 1 if self.links[v[0], 1] == v[1] else 0
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

    def k_exchange(self, v):  # O(n)
        # walk around the tour, check and change the orientation when meet break_vs
        new_links = np.zeros_like(self.links, dtype=int)
        pair2del = {}
        pair2add = {}
        for i in range(len(v)//2):  # O(k)
            pair2del[v[2*i]] = v[2*i+1]
            pair2del[v[2*i+1]] = v[2*i]
            pair2add[v[2*i-1]] = v[2*i]
            pair2add[v[2*i]] = v[2*i-1]
        new_links[v[0], 1] = v[-1]
        new_links[v[-1], 0] = v[0]
        curr_v = v[-1]
        orientation = 0 if self.links[curr_v, 1] == pair2del[curr_v] else 1
        next_v = self.links[curr_v, orientation]
        while True:  # O(n)
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


if __name__ == '__main__':
    test_tour = TourArray(list(range(10)))
    break_vs = [0, 1, 5, 4]
    print(test_tour.check_feasible(break_vs, [0, 1]))
    test_tour.k_exchange(break_vs)
    print(test_tour.route)
