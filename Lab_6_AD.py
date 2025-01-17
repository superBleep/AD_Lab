"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
import networkx as nx
from os import path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyod.models import lof


# Compute hash of graph edge as the sum each nodes' hash
def edgeHash(e):
    return hash(e[0]) + hash(e[1])


# load graph from txt file
def graphLoader(input, output=None, max_rows=None):
    E = np.loadtxt(input, max_rows=max_rows).astype(np.int32)
    E_hashes = np.array([edgeHash(e) for e in E])

    G = nx.Graph()

    weights = {} # {edgeHash: weight}

    for e in E:
        print(f'\rComputing hash for edge {e}...', end='')
        weights[edgeHash(e)] = weights.get(edgeHash(e), 0) + 1
    print('\r' + ' ' * 50, end='\r')
    print('Hashes computed')

    # For each unique edge hash, get its corresponding edge
    # and add it to the graph, together with its weight
    for h in np.unique(E_hashes):
        print(f'\rAdding edge with hash {h}...', end='')
        idx = np.where(E_hashes == h)[0][0]
        e = E[idx]

        G.add_edge(e[0], e[1], weight=weights[h])
    print('\r' + ' ' * 50, end='\r')

    if output != None:
        nx.write_graphml(G, output)

    return G


# --- Exercise 1 ---
def ex1():
    # --- Step 1 ---
    if path.exists('./graph.data'):
        G = nx.read_graphml('./graph.data')
    else:
        G = graphLoader('./ca-AstroPh.txt', './graph.data')

    # --- Step 2 ---
    features = {} # {node: features}

    for n in G.nodes:
        print(f'\rProcessing attributes for node {n}...', end='')
        ego = nx.ego_graph(G, n)

        N_i = len(list(G.neighbors(n)))
        E_i = ego.number_of_edges()
        W_i = sum(e[2] for e in ego.edges.data('weight'))
        
        Aj = nx.to_numpy_array(ego)
        eigvlash, _ = np.linalg.eigh(Aj)
        L_wi = max(eigvlash)

        features[n] = {
            'N': N_i,
            'E': E_i,
            'W': W_i,
            'L': L_wi
        }
    print('\r' + ' ' * 50, end='\r')

    nx.set_node_attributes(G, features)
    print('Features computed and stored in graph')

    # --- Step 3 ---
    N_i = np.array([feature['N'] for feature in features.values()])
    E_i = np.array([feature['E'] for feature in features.values()])

    # Nr. of neighbors in the egonet (X) influences the nr. of edges (Y)
    X = np.log1p(N_i).reshape(-1, 1)
    Y = np.log1p(E_i)

    model = LinearRegression()
    model.fit(X, Y) # Linear regression in log scale

    C = np.exp(model.intercept_) # Get C in real scale
    theta = model.coef_[0]
    Y_pred = C * (N_i ** theta) # Predicted E values, based on the model

    scores = {}
    for i, n in enumerate(G.nodes):
        scores[n] = max(Y[i], Y_pred[i]) / max(Y[i], Y_pred[i]) * np.log1p(np.abs(Y[i] - Y_pred[i]))
    print('Anomaly scores computed')

    # --- Step 4 ---
    scores_desc = sorted(scores.items(), key=lambda data : data[1], reverse=True)
    nodes_top_10 = [np.int32(node) for node, _ in scores_desc[:10]]

    G_2 = graphLoader('./ca-AstroPh.txt', max_rows=1500)

    colors = ['red' if node in nodes_top_10 else 'green' for node in G_2.nodes()]

    plt.figure(1)
    nx.draw(G_2, with_labels=False, node_color=colors, node_size=30, edge_color='gray', alpha=0.7)

    # --- Step 5 ---
    scaler = MinMaxScaler()
    scores_values = np.array(list(scores.values())).reshape(-1, 1)
    scores_norm = scaler.fit_transform(scores_values)

    feature_pairs = np.vstack((N_i, E_i)).T
    lof_model = lof.LOF()
    lof_model.fit(feature_pairs)
    scores_lof = lof_model.decision_scores_

    scores_new = scores_norm.flatten() + scores_lof

    scores_new_full = {}
    for i, n in enumerate(G.nodes):
        scores_new_full[n] = scores_new[i]

    scores_desc = sorted(scores_new_full.items(), key=lambda data : data[1], reverse=True)
    nodes_top_10 = [np.int32(node) for node, _ in scores_desc[:10]]

    colors = ['red' if node in nodes_top_10 else 'green' for node in G_2.nodes()]

    plt.figure(2)
    nx.draw(G_2, with_labels=False, node_color=colors, node_size=30, edge_color='gray', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    ex1()
