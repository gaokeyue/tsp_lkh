from dataclasses import dataclass, field, make_dataclass
import numpy as np
from operator import itemgetter
import time

class CompleteGraph:

    def __init__(self, adj_mat: np.array):
        """The vertices of a graph are set to be (0, 1, 2,...,n-1).
        :param adj_mat -- A dense graph is represented by an n-by-n numpy array adj_mat such that
        adj_mat[i, j] is the weight of edge(i, j). Note that if the graph is directed,
        the direction of edge (i, j) is from i to j. If j is not adjacent to i, then adj_mat[i, j] = np.nan
        By default, adj_mat[i, i] = np.nan
        :param is_complete -- whether the graph is complete.
        """
        self.adj_mat = adj_mat.copy()
        self.n = len(adj_mat)

    def __iter__(self):
        """Iterating over the vertices of the graph"""
        yield from range(self.n)

    def e_weight(self, i, j):
        """Weight of edge(i, j)"""
        return self.adj_mat[i, j]

    def adj(self, i):
        """returns a generator of adjacent vertices of vertex i.
        If is_end=True, then the graph is directed, and adjacency is defined to be
        that there is an edge whose end is i.
        """
        # If the graph is complete, then the adjacent vertices are everything except i
        yield from range(i)
        yield from range(i + 1, self.n)

    @classmethod
    def build_random_complete_graph(cls, n, non_edge_val=0):
        adj_mat = np.random.random((n, n))
        adj_mat += adj_mat.T  # make a symmetric matrix, hence undirected graph
        adj_mat[range(n), range(n)] = non_edge_val
        return cls(adj_mat)


@dataclass(order=True)
class PrimVertex:
    """A vertex class whose attributes are used in Prim's Minimum spanning tree algorithm."""
    id: int = field(compare=False)
    key: float = field(default=np.inf, compare=True)  # The distance to the identified tree component
    parent: int = field(default=None, compare=False)  # parent id in the final MST
    known: bool = field(default=False, compare=False)  #

    def __eq__(self, other):
        return isinstance(other, PrimVertex) and self.id == other.id

def prim_array(graph):
    """Implementation of Prim's Algorithm by array(i.e. python list)"""
    vertices = [PrimVertex(id=i, key=np.inf) for i in range(graph.n)]
    vertices[0].key = 0
    q = list(vertices)
    while len(q) > 0:
        ix, v0 = min(enumerate(q), key=itemgetter(1))
        del q[ix]
        v0.known = True
        for v_id in graph.adj(v0.id):
            v = vertices[v_id]
            w0 = graph.e_weight(v0.id, v.id)
            if not v.known and w0 < v.key:
                v.key = w0
                v.parent = v0.id
        # report the total edge weight of the mst
    result = sum(graph.e_weight(vertex.parent, vertex.id) for vertex in vertices[1:])
    print(f"the total edge weight of the mst is {result}")
    return result

def update_graph_weight(graph, pi):
    d = np.ones((graph.n, graph.n))
    for i in range(graph.n):
        for j in range(i, graph.n):
            d[i][j] = graph.adj_mat[i][j] + pi[i] + pi[j]
            d[j][i] = graph.adj_mat[i][j] + pi[i] + pi[j]
    new_g = CompleteGraph(d)
    return new_g


# weight of m1t after weighted
def weighted_total_weight(graph, pi):
    new_g = update_graph_weight(graph, pi)
    result = build_m1t(new_g)
    w = result[0] - 2 * sum(pi)
    return [w, result[1]]


def build_m1t(graph):  # O(n^2)
    # build minimum 1-tree without assigning the special node from new_g
    # calculate length of minimum 1-tree, degree of each node
    vertices = [PrimVertex(id=i, key=np.inf) for i in range(graph.n)]  # O(n)
    # initial node: 0
    vertices[0].key = 0
    # each node has a parent except node 0
    degree = [1 for _ in range(graph.n)]  # O(n)
    degree[0] = 0
    q = list(vertices)
    tree = []
    while len(q) > 0:  # O(n^2)
        ix, v0 = min(enumerate(q), key=itemgetter(1))
        del q[ix]
        v0.known = True
        tree.append(v0)
        if v0 != vertices[0]:
            degree[v0.parent] += 1
        for v_id in graph.adj(v0.id):  # O(n)
            v = vertices[v_id]
            w0 = graph.e_weight(v0.id, v.id)
            if not v.known and w0 < v.key:
                v.key = w0
                v.parent = v0.id
    # report the total edge weight of the mst
    length = sum(graph.e_weight(vertex.parent, vertex.id) for vertex in vertices[1:])
    # add the edge corresponding to the second nearest neighbor of one of the leaves of the tree
    list_leaves = []
    for i in range(1, graph.n):
        if degree[i] == 1:
            ending_node = nsmallest(3, enumerate(graph.adj_mat[i]), key=itemgetter(1))[2]
            list_leaves.append((ending_node[1], i, ending_node[0]))
    last_edge = max(list_leaves)
    # last_edge[1]: the special node; last_edge[2]: degree += 1; last_edge[0]: the weight of the edge to be added
    degree[last_edge[1]] += 1
    degree[last_edge[2]] += 1
    length += last_edge[0]
    # delete the special node from tree
    special_node = last_edge[1]
    # tree.remove(vertices[special_node])
    return [length, degree, special_node, tree, vertices]


def build_mst(graph):
    # build minimum 1-tree without assigning the special node from graph
    vertices = [PrimVertex(id=i, key=np.inf) for i in range(graph.n)]
    vertices[0].key = 0
    q = list(vertices)
    tree = []
    while len(q) > 0:
        ix, v0 = min(enumerate(q), key=itemgetter(1))
        del q[ix]
        v0.known = True
        tree.append(v0)
        for v_id in graph.adj(v0.id):
            v = vertices[v_id]
            w0 = graph.e_weight(v0.id, v.id)
            if not v.known and w0 < v.key:
                v.key = w0
                v.parent = v0.id
    return tree


# beta_value: the length of the edge to be removed from the spanning tree when edge (i, j) is added
# then alpha(i, j) = e(i, j) - beta(i, j)
def beta(graph):  # O(n^2)
    tree = build_m1t(graph)[3]  # O(n^2)
    special_node = build_m1t(graph)[4]
    n = len(tree)
    beta_value = np.zeros((n, n))
    for i in range(n): # O(n^2)
        if tree[i].id != special_node:
            for j in range(i+1, n):
                if tree[j].id != special_node:
                    value = max(beta_value[tree[i].id][tree[j].parent], graph.e_weight(tree[j].id, tree[j].parent))
                    beta_value[tree[i].id][tree[j].id] = beta_value[tree[j].id][tree[i].id] = value
    return beta_value


def alpha_nearness(graph):  # O(n^2)
    n = graph.n
    special_node = build_m1t(graph)[2]  # O(n^2)
    vertices = build_m1t(graph)[4]
    alpha = np.zeros((n, n))
    beta_value = beta(graph)  # O(n^2)
    for i in range(n):  # O(n^2)
        for j in range(i+1, n):
            if i == special_node or j == special_node:
                list_n = sorted(graph.adj_mat[special_node])
                alpha[i][j] = alpha[j][i] = list_n[2]
            elif i == vertices[j].parent or j == vertices[i].parent:
                alpha[i][j] = alpha[j][i] = 0
            else:
                alpha[i][j] = alpha[j][i] = graph.e_weight(i, j) - beta_value[i][j]
    return alpha


def best_pi(graph):
    pi0 = [0 for _ in range(graph.n)]
    w_0 = weighted_total_weight(graph, pi0)[0]
    degree_0 = weighted_total_weight(graph, pi0)[1]
    v_0 = [d - 2 for d in degree_0]
    v_1 = v_0.copy()
    # calculate the initial step size
    t = 1
    pi = [pi0[i] + t*(0.7*v_0[i]+0.3*v_1[i]) for i in range(graph.n)]
    w_1 = weighted_total_weight(graph, pi)[0]
    while w_1 > w_0:
        t *= 2
        pi = [pi0[i] + t * (0.7 * v_0[i] + 0.3 * v_1[i]) for i in range(graph.n)]
        w_1 = weighted_total_weight(graph, pi)[0]
    t /= 2

    v_optimal = [0 for _ in range(graph.n)]
    pi = [pi0[i] + t * (0.7 * v_0[i] + 0.3 * v_1[i]) for i in range(graph.n)]
    order = graph.n / 2
    while t > 0.0001 and v_0 != v_optimal:
        w_1 = weighted_total_weight(graph, pi)[0]
        degree_1 = weighted_total_weight(graph, pi)[1]
        while True:
            index = 1
            while v_0 != v_optimal and index < order:
                index += 1
                v_1 = v_0.copy()
                v_0 = [d - 2 for d in degree_1]
                pi = [pi[i] + t*(0.7*v_0[i]+0.3*v_1[i]) for i in range(graph.n)]
                degree_1 = weighted_total_weight(graph, pi)[1]
                w_0 = w_1
                w_1 = weighted_total_weight(graph, pi)[0]
            if w_1 >= w_0:
                order /= 2
                break
        t = t/2

    return pi
