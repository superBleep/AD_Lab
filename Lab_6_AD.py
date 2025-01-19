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
import warnings
from scipy.io import loadmat
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch.nn import Module
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# Compute hash of graph edge as the sum each nodes' hash
def edgeHash(e):
    return hash(e[0]) + hash(e[1])


# load graph from txt file
def graphLoader(input, output=None, max_rows=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
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


# First anomaly detection model from Exercise 1
def scorer(G):
    features = {node: data for node, data in G.nodes(data=True)}

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

    return scores


# Anomaly detection model from Exercise 2
def scorer2(G):
    features = {node: data for node, data in G.nodes(data=True)}

    W_i = np.array([feature['W'] for feature in features.values()])
    E_i = np.array([feature['E'] for feature in features.values()])

    # Edge weights of the egonet (X) influence the nr. of edges (Y)
    X = np.log1p(W_i).reshape(-1, 1)
    Y = np.log1p(E_i)

    model = LinearRegression()
    model.fit(X, Y) # Linear regression in log scale

    C = np.exp(model.intercept_) # Get C in real scale
    theta = model.coef_[0]
    Y_pred = C * (W_i ** theta) # Predicted E values, based on the model

    scores = {}
    for i, n in enumerate(G.nodes):
        scores[n] = max(Y[i], Y_pred[i]) / max(Y[i], Y_pred[i]) * np.log1p(np.abs(Y[i] - Y_pred[i]))
    print('Anomaly scores computed')

    return scores


# Second anomaly detection model from Exercise 1
def scorerLOF(G, scores):
    features = {node: data for node, data in G.nodes(data=True)}

    N_i = np.array([feature['N'] for feature in features.values()])
    E_i = np.array([feature['E'] for feature in features.values()])

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

    return scores_new_full


# Second anomaly detection model from Exercise 2
def scorerLOF2(G, scores):
    features = {node: data for node, data in G.nodes(data=True)}

    W_i = np.array([feature['W'] for feature in features.values()])
    E_i = np.array([feature['E'] for feature in features.values()])

    scaler = MinMaxScaler()
    scores_values = np.array(list(scores.values())).reshape(-1, 1)
    scores_norm = scaler.fit_transform(scores_values)

    feature_pairs = np.vstack((W_i, E_i)).T
    lof_model = lof.LOF()
    lof_model.fit(feature_pairs)
    scores_lof = lof_model.decision_scores_

    scores_new = scores_norm.flatten() + scores_lof

    scores_new_full = {}
    for i, n in enumerate(G.nodes):
        scores_new_full[n] = scores_new[i]

    return scores_new_full


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
    scores = scorer(G)


    # --- Step 4 ---
    scores_desc = sorted(scores.items(), key=lambda data : data[1], reverse=True)
    nodes_top_10 = [np.int32(node) for node, _ in scores_desc[:10]]

    G_2 = graphLoader('./ca-AstroPh.txt', max_rows=1500)

    colors = ['red' if node in nodes_top_10 else 'green' for node in G_2.nodes()]

    plt.figure(1)
    nx.draw(G_2, with_labels=False, node_color=colors, node_size=30, edge_color='gray', alpha=0.7)
    plt.show()


    # --- Step 5 ---
    scores = scorerLOF(G, scores)

    scores_desc = sorted(scores.items(), key=lambda data : data[1], reverse=True)
    nodes_top_10 = [np.int32(node) for node, _ in scores_desc[:10]]

    colors = ['red' if node in nodes_top_10 else 'green' for node in G_2.nodes()]

    plt.figure(2)
    nx.draw(G_2, with_labels=False, node_color=colors, node_size=30, edge_color='gray', alpha=0.7)
    plt.show()


# --- Exercise 2 ---
def ex2():
    # --- Step 1 ---
    G_1 = nx.random_regular_graph(3, 100)
    G_2 = nx.connected_caveman_graph(10, 20)

    G_12 = nx.union(G_1, G_2, rename=('g1_', 'g2_'))

    for _ in range(np.random.randint(10, 20)):
        G_12.add_edge(
            np.random.choice([node for node in list(G_12.nodes) if node.startswith('g1')]),
            np.random.choice([node for node in list(G_12.nodes) if node.startswith('g2')])
        )

    features = {} # {node: features}

    for n in G_12.nodes:
        print(f'\rProcessing attributes for node {n}...', end='')
        ego = nx.ego_graph(G_12, n)

        N_i = len(list(G_12.neighbors(n)))
        E_i = ego.number_of_edges()

        features[n] = {
            'N': N_i,
            'E': E_i
        }
    print('\r' + ' ' * 50, end='\r')

    nx.set_node_attributes(G_12, features)
    print('Features computed and stored in graph')

    scores = scorer(G_12)
    scores = scorerLOF(G_12, scores)

    scores_desc = sorted(scores.items(), key=lambda data : data[1], reverse=True)
    nodes_top_10 = [node for node, _ in scores_desc[:10]]

    colors = ['red' if node in nodes_top_10 else 'green' for node in G_12.nodes()]

    plt.figure(1)
    nx.draw(G_12, with_labels=False, node_color=colors, node_size=30, edge_color='gray', alpha=0.7)
    plt.show()
    

    # --- Step 2 ---
    G_3 = nx.random_regular_graph(3, 100)
    G_4 = nx.random_regular_graph(5, 100)

    G_34 = nx.union(G_3, G_4, rename=('g3_', 'g4_'))

    for e in G_34.edges():
        G_34.add_edge(e[0], e[1], weight=1)

    for n in np.random.choice(list(G_34.nodes), 2):
        ego = nx.ego_graph(G_34, n)

        for e in ego.edges().data():
            G_34[e[0]][e[1]]['weight'] += 10

    features = {} # {node: features}

    for n in G_34.nodes:
        print(f'\rProcessing attributes for node {n}...', end='')
        ego = nx.ego_graph(G_34, n)

        W_i = sum(e[2] for e in ego.edges.data('weight'))
        E_i = ego.number_of_edges()

        features[n] = {
            'W': W_i,
            'E': E_i
        }
    print('\r' + ' ' * 50, end='\r')

    nx.set_node_attributes(G_34, features)
    print('Features computed and stored in graph')

    scores = scorer2(G_34)
    scores = scorerLOF2(G_34, scores)

    scores_desc = sorted(scores.items(), key=lambda data : data[1], reverse=True)
    nodes_top_10 = [node for node, _ in scores_desc[:4]]

    colors = ['red' if node in nodes_top_10 else 'green' for node in G_34.nodes()]

    plt.figure(2)
    nx.draw(G_34, with_labels=False, node_color=colors, node_size=30, edge_color='gray', alpha=0.7)
    plt.show()


# Encoder from Exercise 3
class Encoder(Module):
    def __init__(self, in_chan):
        super.__init__(self)

        self.conv1 = GCNConv(in_chan, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
    

# Attribute decoder from Exercise 3
class AttributeDecoder(Module):
    def __init__(self, out_chan):
        super.__init__(self)

        self.conv1 = GCNConv(64, 128)
        self.conv2 = GCNConv(128, out_chan)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
    

# Structure decoder from Exercise 3
class StructureDecoder(Module):
    def __init__(self):
        super.__init__(self)

        self.conv1 = GCNConv(64, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x @ x.T


# Graph autoencoder from Exercise 3
class GAE(Module):
    def __init__(self, n):
        super.__init__(self)

        self.encoder = Encoder(n)
        self.att_decoder = AttributeDecoder(n)
        self.struc_decoder = StructureDecoder()

    def forward(self, x):
        x = self.encoder(x)
        # ???


# --- Exercise 3 ---
def ex3():
    # --- Step 2 ---
    data = loadmat('./ACM.mat')
    att, adj, labels = data['Attributes'], data['Network'], data['Label']
    edges_idx, edges_att = from_scipy_sparse_matrix(adj)

    autoencoder = GAE()
    autoencoder.fit()


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()