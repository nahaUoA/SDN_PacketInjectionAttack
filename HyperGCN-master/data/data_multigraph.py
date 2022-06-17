import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle


def load_data(data_arg):
    """
    parses the dataset
    """
    dataset = parser(data_arg['data'], data_arg['dataset'], data_arg['usage']).parse()

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, data_arg['data'], data_arg['usage'], data_arg['dataset'], data_arg['usage'] + ".pickle")

    if not os.path.isfile(file): print(data_arg['usage'] + ".pickle does not exist")
    with open(file, 'rb') as H:
        usage = pickle.load(H)      # either train or test set

    return dataset, usage



class parser(object):
    """
    an object for parsing data
    """

    def __init__(self, data, dataset, usage):
        """
        initialises the data directory

        arguments:
        data: coauthorship/cocitation
        dataset: cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
        """

        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, data, usage, dataset)
        self.data, self.dataset, self.usage = data, dataset, usage

    def parse(self):
        """
        returns a dataset specific function to parse
        """

        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()

    def _load_data(self):
        """
        loads the coauthorship hypergraph, features, and labels of cora

        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: coauthorship hypergraph
        features.pickle: bag of word features
        labels.pickle: labels of papers

        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels as keys
        """

        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)
            # print(hypergraph)
            print("number of hyperedges is", len(hypergraph))

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()
            # features = pickle.load(handle)
            # print(features.shape)

        with open(os.path.join(self.d, 'label_powerset.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle), num_class=32)
            # labels = pickle.load(handle)
            # print(type(labels))
            # print(labels)
            # print(labels.shape)

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}

    def _1hot(self, labels, num_class):
        """
        converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
        labels: a list of positive integers with eah integer representing a unique label
        """

        classes = set(c for c in range(num_class))
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)