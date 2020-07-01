import numpy as np
import time
from tsp_lkh_zy.prim import CompleteGraph, PrimVertex
from operator import itemgetter
from heapq import nsmallest


def update_graph_weight(graph, pi):
    # d = np.ones((graph.n, graph.n))
    # for i in range(graph.n):
    #     for j in range(i, graph.n):
    #         d[i][j] = graph.adj_mat[i][j] + pi[i] + pi[j]
    #         d[j][i] = graph.adj_mat[i][j] + pi[i] + pi[j]
    new_g = CompleteGraph(graph.adj_mat + pi.reshape(graph.n, -1) + pi.reshape(-1, graph.n))
    return new_g


# weight of m1t after weighted
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
    print(f"The total weight of mst is {length}")
    # add the edge corresponding to the second nearest neighbor of one of the leaves of the tree
    list_leaves = []
    for i in range(graph.n):
        if degree[i] == 1:
            ending_node = nsmallest(3, enumerate(graph.adj_mat[i]), key=itemgetter(1))[1]
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


def get_alpha_nearness(graph):  # O(n^2)
    n = graph.n
    m1t_result =  build_m1t(graph)
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


# def best_pi(graph):
#     pi0 = [0 for _ in range(graph.n)]
#     w_0 = weighted_total_weight(graph, pi0)[0]
#     degree_0 = weighted_total_weight(graph, pi0)[1]
#     v_0 = [d - 2 for d in degree_0]
#     v_1 = v_0.copy()
#     # calculate the initial step size
#     t = 1
#     pi = [pi0[i] + t * (0.7 * v_0[i] + 0.3 * v_1[i]) for i in range(graph.n)]
#     w_1 = weighted_total_weight(graph, pi)[0]
#     while w_1 > w_0:
#         t *= 2
#         pi = [pi0[i] + t * (0.7 * v_0[i] + 0.3 * v_1[i]) for i in range(graph.n)]
#         w_1 = weighted_total_weight(graph, pi)[0]
#     t /= 2
#
#     v_optimal = [0 for _ in range(graph.n)]
#     pi = [pi0[i] + t * (0.7 * v_0[i] + 0.3 * v_1[i]) for i in range(graph.n)]
#     order = graph.n / 2
#     while t > 0.0001 and v_0 != v_optimal:
#         w_1, degree_1 = weighted_total_weight(graph, pi)
#         while True:
#             index = 1
#             while v_0 != v_optimal and index < order:
#                 index += 1
#                 v_1 = v_0.copy()
#                 v_0 = [d - 2 for d in degree_1]
#                 pi = [pi[i] + t * (0.7 * v_0[i] + 0.3 * v_1[i]) for i in range(graph.n)]
#                 degree_1 = weighted_total_weight(graph, pi)[1]
#                 w_0 = w_1
#                 w_1 = weighted_total_weight(graph, pi)[0]
#             if w_1 >= w_0:
#                 order /= 2
#                 break
#         t = t / 2
#
#     return pi


def dual_ascent(graph, eps=10 ** -6):
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
            print(f"step size={t}, objective={w1:.3f}")
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
    print(f"First period done, objective is {w0}")
    # Rest of the periods
    max_iter = 512
    n_iter = 0
    while t >= eps and n_iter < max_iter:
        while True:  # a period can be executed for indefinitely many times.
            print(f"Start period={period}, step size={t}, w0={w0:.4f}")
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
                print(f"End period={period}")
                period = (period + 1) // 2
                t /= 2
                w0 = w1
                break
            else:
                if (w1 - w0) / w0 < eps:
                    return pi_star, w_star, grad_star
                print(f"Continue period={period}")
                w0 = w1
    print(f"Total number of iteration is {n_iter}")
    return pi_star, w_star, grad_star


if __name__ == '__main__':
    cost_mat= np.load('../data/ch130.npy')
    # graph = CompleteGraph(cost_mat)
    # print(f"The initial objective is {weighted_total_weight(graph, np.zeros(130))[0]}")
    # t0 = time.perf_counter()
    # pi_star, w_star, grad_star = dual_ascent(graph)
    # elapsed = time.perf_counter() - t0
    # print(f"It took {elapsed:.6f} seconds and got objective={w_star}, \n grad={grad_star}")
    from scipy.sparse.csgraph import minimum_spanning_tree
    sp_tree = minimum_spanning_tree(cost_mat)
