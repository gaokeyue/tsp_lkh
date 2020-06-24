import json
import collections
from operator import itemgetter
import numpy as np
from prim import CompleteGraph
from search_for_d import best_pi, alpha_nearness, weighted_total_weight, update_graph_weight, build_m1t

with open('20190101010101-001-Stations.json', 'r', encoding='utf-8') as f:
    stations = json.loads(f.read())

n = len(stations)
areas = []
for i in range(n):
    areas.append(stations[i]["area_id"])
areas = set(areas)
areas = list(areas)

partition_graph = collections.defaultdict(list)

for area in areas:
    for item in stations:
        if item['area_id'] == area:
            partition_graph[area].append(item['id'])

lengths = [len(item) for item in partition_graph.values()]
ix, length = max(enumerate(lengths), key=itemgetter(1))
nodes = list(partition_graph.values())[ix]

with open('20190101010101-001-MapMatrix.json', 'r') as f:
    distance_data = json.load(f)

distance_mat = np.zeros((len(nodes), len(nodes)))
for i in range(len(nodes)):
    for j in range(len(nodes)):
        distance_mat[i][j] = distance_data[nodes[i]][nodes[j]]['distance']


graph1 = CompleteGraph(distance_mat)
pi0 = [0 for _ in range(graph1.n)]
list_w = best_pi(graph1)
# print(list_w)
# print(weighted_total_weight(graph1, pi0))
# print(alpha_nearness(graph1))
# print(weighted_total_weight(graph1, list_w))
# print(alpha_nearness(update_graph_weight(graph1, list_w)))

'''
s = [0, 3, 9, 1, 999, 8]
a = [3, 0, 6, 999, 999, 999]
b = [9, 6, 0, 999, 999, 999]
c = [1, 999, 999, 0, 7, 999]
d = [999, 999, 999, 7, 0, 2]
e = [8, 999, 999, 999, 2, 0]
mat = np.array([s, a, b, c, d, e])
print(mat)
graph2 = CompleteGraph(mat)
best = best_pi(graph2)
pi1 = [0 for _ in range(graph2.n)]
print(best)
print(weighted_total_weight(graph2, pi1))
print(alpha_nearness(graph2))
print(weighted_total_weight(graph2, best))
print(alpha_nearness(update_graph_weight(graph2, best)))
tree = build_m1t(update_graph_weight(graph2, best))[3]
special = build_m1t(update_graph_weight(graph2, best))[2]
print([item.id for item in tree])
print(special)
'''
