"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
import networkx as nx


def edgeHash(e):
    return hash(e[0]) + hash(e[1])


# Exercise 1
def ex1():
    E = np.loadtxt("./ca-AstroPh.txt").astype(np.int32)
    E = tuple(map(tuple, E))
    hashes = np.array([edgeHash(e) for e in E])
    G = nx.Graph()

    weights = {}

    for e in E:
        if e not in weights:
            weights[edgeHash(e)] = 1
        else:
            weights[edgeHash(e)] += 1

    for h in np.unique(hashes):
        idx = np.where(hashes == h)[0][0]
        e = E[idx]

        G.add_edge(e[0], e[1], weight=weights[h])

    print(G)

if __name__ == '__main__':
    ex1()
