import numpy as np
from tsp_lkh_zy.prim import CompleteGraph, PrimVertex
from operator import itemgetter
from heapq import nsmallest


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
    # new_g = update_graph_weight(graph, pi)

    new_g = CompleteGraph(graph.adj_mat + pi.reshape(graph.n, -1) + pi.reshape(-1, graph.n))
    length, degree, _, _, _ = build_m1t(new_g)
    w = length - 2 * sum(pi)
    return w, np.array(degree)


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
    pi0 = np.zeros(graph.n)
    w_0, degree_0 = weighted_total_weight(graph, pi0)
    print(f"w0={w_0}")
    v_0 = degree_0 - 2
    v_1 = v_0.copy()
    # calculate the initial step size
    t = 1
    pi = pi0 + t*(0.7*v_0+0.3*v_1)
    w_1, degree_1 = weighted_total_weight(graph, pi)
    while w_1 > w_0:
        t *= 2
        pi = pi0 + t * (0.7 * v_0 + 0.3 * v_1)
        w_1, degree_1 = weighted_total_weight(graph, pi)
    t /= 2
    print(f"Initial step size={t}, w={w_1}")

    pi = pi0 + t * (0.7 * v_0 + 0.3 * v_1)
    order = graph.n / 2
    while t > 0.0001 and v_0.any():
        w_1, degree_1 = weighted_total_weight(graph, pi)
        print(f"step size={t}, w={w_1}")
        while True:
            index = 1
            while v_0.any() and index < order:
                index += 1
                v_1 = v_0.copy()
                v_0 = degree_1 - 2
                pi = pi + t*(0.7*v_0+0.3*v_1)
                w_0 = w_1
                w_1, degree_1 = weighted_total_weight(graph, pi)
            if w_1 >= w_0:
                order /= 2
                break
        t = t/2
    print(v_0)
    return pi


if __name__ == '__main__':
    cost_mat = np.load('../data/ch130.npy')
    n = len(cost_mat)
    # cost_mat[range(n), range(n)] = np.inf
    graph = CompleteGraph(cost_mat)
    my_bestpi = best_pi(graph)
