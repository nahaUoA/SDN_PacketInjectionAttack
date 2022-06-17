import networkx as nx
import os, inspect, shutil
import random
import numpy as np, scipy.sparse as sp
import pickle
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def extract_hypergraph(G):
    hypergraph = {}
    for edge_id, edge in enumerate(G.edges()):
        hypergraph[edge_id] = set()
        hypergraph[edge_id].add(int(edge[0]))
        hypergraph[edge_id].add(int(edge[1]))

    return hypergraph




def extract_features(G):
    attributes_names = ['id','Internal','Latitude', 'Longitude']
    # 'Country' and 'Label' is not included because they are string
    V, p = G.number_of_nodes(), len(attributes_names)

    features = np.zeros((V,p), dtype=float)

    for i, name in enumerate(attributes_names):
        attribute = nx.get_node_attributes(G, name)
        for nodeid in range(V):
            if str(nodeid) in attribute.keys():
                features[nodeid][i] = attribute[str(nodeid)]
            else:
                features[nodeid][i] = 0

    features = sp.csr_matrix(features)

    return features




def extract_labels(G):
    # original labels, documented within each nodes
    # V = G.number_of_nodes()
    #
    # string_labels = []  # ['Sydney1', 'Brisbane2', 'Canberra1', .... , 'Darwin']
    # for label in nx.get_node_attributes(G, 'label').values():
    #     if label not in string_labels:
    #         string_labels.append(label)
    #
    #
    # labels = [-1 for _ in range(V)]
    # for nodeid in range(V):
    #     for i, label in enumerate(string_labels):
    #         if nx.get_node_attributes(G, 'label')[str(nodeid)] == label:
    #             labels[nodeid] = i

    # new labels, after discussed with Dr. Hung

    V = G.number_of_nodes()
    labels = [G.graph['Backbone'] for nodeid in range(V)]

    return labels

def bin2dec(binary):
    return int("".join(str(x) for x in binary), 2)


def extract_multilabels(G):
    label_names = ['Access', 'Backbone', 'Commercial', 'Customer', 'Transit']
    V, q = G.number_of_nodes(), len(label_names)
    multilabels = [[-1 for _ in range(q)] for __ in range(V)]

    for i, name in enumerate(label_names):
        if name not in G.graph.keys():
            label = 0
        else:
            label = G.graph[name]
        for nodeid in range(V):
            multilabels[nodeid][i] = label


    return multilabels


def extract_label_powerset(G):
    V = G.number_of_nodes()
    multilabels = extract_multilabels(G)

    label_powerset = [bin2dec(multilabels[nodeid]) for nodeid in range(V)]

    return np.array(label_powerset)


def dataset_splits(G):
    splits = {}
    V = G.number_of_nodes()
    nodes = list(range(V))
    random.shuffle(nodes)
    # train_size = round(0.75*V)
    # test_size = V - train_size

    train_size = 0
    test_size = V - train_size
    train_set, test_set = train_test_split(nodes, train_size=train_size, test_size=test_size)
    splits['train'] = train_set
    splits['test'] = test_set

    return splits


def generate_nodes(G):
    return [v for v in range(G.number_of_nodes())]


def delete_folder(path):
    folder = path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def graph_processing(G):
    hypergraph = extract_hypergraph(G)
    features = extract_features(G)
    multilabels = extract_multilabels(G)
    label_powerset = extract_label_powerset(G)


    return hypergraph, features, multilabels, label_powerset


def main():
    data = "internet_topology_zoo"
    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    raw_datapath = os.path.join(Dir, "raw_data", data)
    files = [f for f in os.listdir(raw_datapath)
                if (os.path.isfile(os.path.join(raw_datapath, f)) and f[-8:] == '.graphml')]
    train_networks = random.sample(files, round(0.3 * len(files)))
    test_networks = [f for f in files if f not in train_networks]


    # generating training files
    train_datapath = os.path.join(Dir, "data", data, "train")
    delete_folder(train_datapath)
    for f in train_networks:
        # print(os.path.join(train_datapath,f[:-8]))
        train_dir = os.path.join(train_datapath, f[:-8])
        os.mkdir(train_dir)

        G = nx.read_graphml(os.path.join(raw_datapath, f))
        hypergraph, features, multilabels, label_powerset = graph_processing(G)
        nodes = generate_nodes(G)

        with open(os.path.join(train_dir, "train.pickle"), "wb") as train_out:
            pickle.dump(nodes, train_out)
            train_out.close()
        with open(os.path.join(train_dir, "hypergraph.pickle"), "wb") as hypergraph_out:
            pickle.dump(hypergraph, hypergraph_out)
            hypergraph_out.close()
        with open(os.path.join(train_dir, "features.pickle"), "wb") as features_out:
            pickle.dump(features, features_out)
            features_out.close()
        with open(os.path.join(train_dir, "multilabels.pickle"), "wb") as multilabels_out:
            pickle.dump(multilabels, multilabels_out)
            multilabels_out.close()
        with open(os.path.join(train_dir, "label_powerset.pickle"), "wb") as labels_out:
            pickle.dump(label_powerset, labels_out)
            labels_out.close()

    # generating test files
    test_datapath = os.path.join(Dir, "data", data, "test")
    delete_folder(test_datapath)
    for f in test_networks:
        test_dir = os.path.join(test_datapath, f[:-8])
        os.mkdir(test_dir)

        G = nx.read_graphml(os.path.join(raw_datapath, f))
        hypergraph, features, multilabels, label_powerset = graph_processing(G)
        nodes = generate_nodes(G)

        with open(os.path.join(test_dir, "test.pickle"), "wb") as test_out:
            pickle.dump(nodes, test_out)
            test_out.close()
        with open(os.path.join(test_dir, "hypergraph.pickle"), "wb") as hypergraph_out:
            pickle.dump(hypergraph, hypergraph_out)
            hypergraph_out.close()
        with open(os.path.join(test_dir, "features.pickle"), "wb") as features_out:
            pickle.dump(features, features_out)
            features_out.close()
        with open(os.path.join(test_dir, "multilabels.pickle"), "wb") as multilabels_out:
            pickle.dump(multilabels, multilabels_out)
            multilabels_out.close()
        with open(os.path.join(test_dir, "label_powerset.pickle"), "wb") as labels_out:
            pickle.dump(label_powerset, labels_out)
            labels_out.close()


if __name__ == "__main__":
    main()

