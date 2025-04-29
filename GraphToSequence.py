import numpy as np
from operator import itemgetter

import networkx as nx
import matplotlib.pyplot as plt


def edge_count(edges):
    unique, counts = np.unique(edges, return_counts=True)
    return np.array(unique), np.array(counts)


def sequence_subgraph(connNodes, nodeDegree, alledges, edges, visited=np.array([]), done=np.array([])):
    seq = ""

    while (edges.shape[0] > 0):
        _, indx, _ = np.intersect1d(connNodes, np.setdiff1d(edges, visited, assume_unique=True), assume_unique=True, return_indices=True)
        v = (connNodes[indx])[np.argmin(nodeDegree[indx])]
        seq += str(v) + "."

        opts = np.argwhere(alledges == v)
        edges = alledges[opts[:, 0]-1, opts[:, 1]]

        if (len(visited) > 1):
            for edge in np.intersect1d(edges, visited[:-1]):
                seq += "*" + str(edge) + "."

        visited = np.append(visited, v)
        edges = np.setdiff1d(edges, visited, assume_unique=True)

        if (edges.shape[0] > 1):
            for edge in edges:
                if edge not in visited:
                    seq += "("
                    new_seq, edges, visited, done = sequence_subgraph(connNodes, nodeDegree, alledges, np.array([edge]), visited, done)
                    seq += new_seq
                    seq += ")"

        if (edges.shape[0] <= 1):
            np.append(done, v)

    return seq, edges, visited, done


def sequence(alledges):
    seq = ""
    connNodes, nodeDegree = edge_count(alledges)

    visited = np.array([], dtype=np.int32)
    done = np.array([], dtype=np.int32)
    edges = np.array([], dtype=np.int32)

    while (not (np.isin(connNodes, visited)).all()):
        if (edges.shape[0] > 0):
            _, indx, _ = np.intersect1d(connNodes, np.setdiff1d(edges, visited, assume_unique=True), assume_unique=True, return_indices=True)
            v = (connNodes[indx])[np.argmin(nodeDegree[indx])]
        else:
            indx = np.isin(connNodes, visited, assume_unique=True, invert=True)
            v = (connNodes[indx])[np.argmin(nodeDegree[indx])]
        seq += str(v) + "."

        print(v)
        opts = np.argwhere(alledges == v)
        edges = alledges[opts[:, 0]-1, opts[:, 1]]

        if (len(visited) > 1):
            for edge in np.intersect1d(edges, visited[:-1]):
                seq += "*" + str(edge) + "."

        visited = np.append(visited, v)
        edges = np.setdiff1d(edges, visited, assume_unique=True)

        if (edges.shape[0] > 1):
            for edge in edges:
                if edge not in visited:
                    seq += "("
                    new_seq, edges, visited, done = sequence_subgraph(connNodes, nodeDegree, alledges, np.array([edge]), visited, done)
                    seq += new_seq
                    seq += ")"

        if (edges.shape[0] <= 1):
            np.append(done, v)

        if (edges.shape[0] == 0):
            seq += ";"

    return seq


if __name__ == "__main__":
    nodes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int64)
    # edges = np.array([[1, 2], [2, 11], [11, 13], [11, 12], [12, 6], [6, 7], [7, 9], [9, 10], [10, 5], [5, 6], [5, 3], [3, 1], [8, 4]], dtype=np.int64).T
    edges = np.array([[1, 2], [2, 11], [11, 13], [11, 12], [12, 6], [7, 9], [9, 10], [10, 5], [5, 3], [3, 1], [8, 4]], dtype=np.int64).T

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.T)

    nx.draw(G, with_labels=True)

    print(sequence(edges))
    plt.show()
