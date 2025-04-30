import numpy as np
import re

import networkx as nx
import matplotlib.pyplot as plt


def edge_count(edges):
    unique, counts = np.unique(edges, return_counts=True)
    return np.array(unique), np.array(counts)


def sequence_subgraph(connNodes, nodeDegree, alledges, edge, last, visited=np.array([]), done=np.array([])):
    seq = ""
    edges = np.array([edge])

    while (edges.shape[0] > 0):
        _, indx, _ = np.intersect1d(connNodes, np.setdiff1d(edges, visited, assume_unique=True), assume_unique=True, return_indices=True)
        v = (connNodes[indx])[np.argmin(nodeDegree[indx])]
        seq += str(v) + "."

        opts = np.argwhere(alledges == v)
        edges = alledges[opts[:, 0]-1, opts[:, 1]]

        if (len(visited) > 1):
            for edge in np.intersect1d(edges, visited[:-1]):
                if (edge != last):
                    seq += "*" + str(edge) + "."

        visited = np.append(visited, v)
        edges = np.setdiff1d(edges, visited, assume_unique=True)

        if (edges.shape[0] > 1):
            for edge in edges:
                if edge not in visited:
                    seq += "("
                    new_seq, edges, visited, done = sequence_subgraph(connNodes, nodeDegree, alledges, edge, v, visited, done)
                    seq += new_seq
                    seq += ")"

        if (edges.shape[0] <= 1):
            np.append(done, v)
        last = v

    return seq, edges, visited, done


def graphToSequence(alledges):
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
                    new_seq, edges, visited, done = sequence_subgraph(connNodes, nodeDegree, alledges, edge, v, visited, done)
                    seq += new_seq
                    seq += ")"

        if (edges.shape[0] <= 1):
            np.append(done, v)

        if (edges.shape[0] == 0):
            seq += ";"

    return seq


def subgraph(checks):
    nodes = []
    edges = []

    if (checks[0] == ""):
        return nodes, edges, checks

    node = int(checks[0])
    nodes.append(node)
    i = 1
    while i < len(checks):
        if (")" in checks[i] or checks[i] == ""):
            checks[i] = checks[i].replace(")", "", 1)
            if (checks[i] != ""):
                return nodes, edges, checks[i-1:]
            return nodes, edges, checks[i:]
        next = int(re.sub('[*()]', '', checks[i]))

        edges.append([node, next])

        if ("(" in checks[i]):
            checks[i] = checks[i].replace("(", "", 1)
            subnodes, subedges, checks = subgraph(checks[i:])
            i = 0
            nodes.extend(subnodes)
            edges.extend(subedges)
        elif ("*" not in checks[i]):
            node = next
            nodes.append(node)

        i += 1

    return nodes, edges, []


def sequenceToGraph(seq):
    nodes = []
    edges = []

    for subseq in seq.split(";"):
        subnodes, subedges, _ = subgraph(subseq.split("."))
        nodes.extend(subnodes)
        edges.extend(subedges)

    return np.array(nodes), np.array(edges).T


if __name__ == "__main__":
    nodes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int64)
    edges = np.array([[1, 2], [2, 11], [11, 13], [11, 12], [12, 6], [6, 7], [7, 9], [9, 10], [10, 5], [5, 6], [5, 3], [3, 1], [8, 4]], dtype=np.int64).T
    # edges = np.array([[1, 2], [2, 11], [11, 13], [11, 12], [12, 6], [7, 9], [9, 10], [10, 5], [5, 3], [3, 1], [8, 4]], dtype=np.int64).T

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.T)

    nx.draw_circular(G, with_labels=True)
    plt.show()

    seq = graphToSequence(edges)
    print(seq)

    res_nodes, res_edges = sequenceToGraph(seq)
    print(res_nodes)
    print(res_edges)

    resG = nx.Graph()
    resG.add_nodes_from(res_nodes)
    resG.add_edges_from(res_edges.T)

    print(nx.graph_edit_distance(G, resG))

    nx.draw_circular(resG, with_labels=True)
    plt.show()
