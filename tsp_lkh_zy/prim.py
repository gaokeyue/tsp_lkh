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

if __name__ == '__main__':
    t0 = time.perf_counter()
    graph = CompleteGraph.build_random_complete_graph(5000)
    prim_array(graph)
    t1 = time.perf_counter()
    print(f"{t1-t0} seconds")