import numpy as np
from dataclasses import dataclass, field, make_dataclass
from operator import itemgetter
import heapq


class CompleteGraph:

    def __init__(self, adj_mat: np.array):
        """The vertices of a graph are set to be (0, 1, 2,...,n-1).
        :param adj_mat -- A dense graph is represented by an n-by-n numpy array adj_mat such that
        adj_mat[i, j] is the weight of edge(i, j). Note that if the graph is directed,
        the direction of edge (i, j) is from i to j. If j is not adjacent to i, then adj_mat[i, j] = np.nan
        By default, adj_mat[i, i] = np.inf
        :param is_complete -- whether the graph is complete.
        """
        self.n = len(adj_mat)
        self.adj_mat = adj_mat.copy()
        self.adj_mat[range(self.n), range(self.n)] = np.inf

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


def weighted_total_weight(graph, pi):
    new_g = CompleteGraph(graph.adj_mat + pi.reshape(graph.n, -1) + pi.reshape(-1, graph.n))
    m1t_result = build_m1t(new_g)
    length = m1t_result['length']
    degree = m1t_result['degree']
    w = length - 2 * sum(pi)
    return w, degree - 2


def build_m1t(graph):  # O(n^2)
    # build minimum 1-tree without assigning the special node from new_g
    # calculate length of minimum 1-tree, degree of each node
    vertices = [PrimVertex(id=i, key=np.inf) for i in range(graph.n)]  # O(n)
    # initial node: 0
    vertices[0].key = 0
    # each node has a parent except node 0
    degree = np.ones(graph.n, dtype=int)  # O(n)
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
    for i in range(graph.n):
        if degree[i] == 1:
            ending_node = heapq.nsmallest(3, enumerate(graph.adj_mat[i]), key=itemgetter(1))[1]
            list_leaves.append((ending_node[1], i, ending_node[0]))
    last_edge = max(list_leaves)
    # last_edge[1]: the special node; last_edge[2]: degree += 1; last_edge[0]: the weight of the edge to be added
    degree[last_edge[1]] += 1
    degree[last_edge[2]] += 1
    length += last_edge[0]
    # delete the special node from tree
    special_node = last_edge[1]
    # tree.remove(vertices[special_node])
    result = {'length': length,
              'degree': degree,
              'special_node': special_node,
              'tree': tree,
              'vertices': vertices,
              'special_edges': (vertices[special_node].parent, last_edge[2])
              }
    return result


def dual_ascent(cost_mat, eps=10 ** -6, max_iter=512):
    graph = CompleteGraph(cost_mat)
    # First period
    period = graph.n // 2
    t = 1
    pi0 = np.zeros(graph.n)
    w0, grad0 = weighted_total_weight(graph, pi0)
    pi1 = pi0 + t * grad0
    pi_star = pi0
    w_star = w0
    grad_star = grad0
    for _ in range(period):
        w1, grad1 = weighted_total_weight(graph, pi1)
        if w1 > w0:
            pi_star = pi1
            w_star = w1
            grad_star = grad1
        if grad1.any():  # since grad1 is of integer type
            # print(f"step size={t}, objective={w1:.3f}")
            if w1 <= w0:
                t /= 2
                pi1 = pi0 + t * (.7 * grad1 + .3 * grad0)  # recompute pi1 using a smaller step size
                break
            t *= 2
            pi0 = pi1
            pi1 = pi0 + t * (.7 * grad1 + .3 * grad0)
            w0 = w1
            grad0 = grad1
        else:
            return pi_star, w_star, grad_star
    # print(f"First period done, objective is {w0}")
    # Rest of the periods
    n_iter = 0
    while t >= eps and n_iter < max_iter:
        while True:  # a period can be executed for indefinitely many times.
            # print(f"Start period={period}, step size={t}, w0={w0:.4f}")
            for _ in range(period):
                n_iter += 1
                if n_iter > max_iter:
                    print(f"Exceed maximum iteration {max_iter}")
                    return pi_star, w_star, grad_star
                w1, grad1 = weighted_total_weight(graph, pi1)
                if w1 > w0:
                    pi_star = pi1
                    w_star = w1
                    grad_star = grad1
                if grad1.any():  # since grad1 is of integer type
                    pi1 = pi1 + t * (.7 * grad1 + .3 * grad0)
                    grad0 = grad1
                else:
                    return pi_star, w_star, grad_star
            if w1 <= w0:
                # print(f"End period={period}")
                period = (period + 1) // 2
                t /= 2
                w0 = w1
                break
            else:
                if (w1 - w0) / w0 < eps:
                    return pi_star, w_star, grad_star
                # print(f"Continue period={period}")
                w0 = w1
    print(f"Total number of iteration is {n_iter}")
    return pi_star, w_star, grad_star


def get_beta(graph, tree, special_node):  # O(n^2)
    n = len(tree)
    beta_value = np.zeros((n, n))
    for i in range(n):  # O(n^2)
        if tree[i].id != special_node:
            for j in range(i + 1, n):
                if tree[j].id != special_node:
                    value = max(beta_value[tree[i].id][tree[j].parent], graph.e_weight(tree[j].id, tree[j].parent))
                    beta_value[tree[i].id][tree[j].id] = beta_value[tree[j].id][tree[i].id] = value
    return beta_value


def get_alpha_nearness(cost_mat):  # O(n^2)
    graph = CompleteGraph(cost_mat)
    n = graph.n
    m1t_result = build_m1t(graph)
    # length = m1t_result['length']
    # degree = m1t_result['degree']
    special_node = m1t_result['special_node']
    special_edges = m1t_result['special_edges']
    tree = m1t_result['tree']
    vertices = m1t_result['vertices']
    alpha = np.zeros((n, n))
    alpha[range(n), range(n)] = np.nan
    beta_value = get_beta(graph, tree, special_node)  # O(n^2)
    for j in range(n):
        if j not in special_edges and j != special_node:
            alpha[special_node, j] = graph.e_weight(special_node, j) - graph.e_weight(special_node, special_edges[1])
    for i in range(n):  # O(n^2)
        for j in range(i + 1, n):
            if i != special_node and j != special_node:
                if i != vertices[j].parent and j != vertices[i].parent:
                    alpha[i][j] = alpha[j][i] = graph.e_weight(i, j) - beta_value[i][j]
    return alpha
