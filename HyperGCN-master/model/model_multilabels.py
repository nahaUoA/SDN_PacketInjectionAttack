from model import networks_multilabels
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
from sklearn.metrics import accuracy_score

from torch.autograd import Variable
from tqdm import tqdm
from model import utils


def train(HyperGCN, dataset, T, ite, args):
    """
    train for a certain number of epochs

    arguments:
	HyperGCN: a dictionary containing model details (gcn, optimiser)
	dataset: the entire dataset
	T: training indices
	args: arguments

	returns:
	the trained model
    """

    hypergcn, optimiser = HyperGCN['model'], HyperGCN['optimiser']
    hypergcn.train()

    X, Y = dataset['features'], dataset['labels']

    for epoch in tqdm(range(args.epochs)):
        if ite == 0:
            optimiser.zero_grad()
        Z = hypergcn(X)
        # print(Z.shape)
        loss = F.binary_cross_entropy(Z[T], Y[T])
        loss.backward()
        optimiser.step()

    HyperGCN['model'] = hypergcn
    HyperGCN['optimiser'] = optimiser
    # HyperGCN['loss'] = loss
    return HyperGCN


def test(HyperGCN, dataset, t, args):
    """
    test HyperGCN

    arguments:
	HyperGCN: a dictionary containing model details (gcn)
	dataset: the entire dataset
	t: test indices
	args: arguments

	returns:
	accuracy of predictions
    """

    hypergcn = HyperGCN['model']
    hypergcn.eval()
    X, Y = dataset['features'], dataset['labels']

    Z = hypergcn(X)

    return accuracy(Z[t], Y[t])


def accuracy(Z, Y):
    """
    arguments:
    Z: predictions
    Y: ground truth labels

    returns:
    accuracy
    """

    predictions = Z.round()
    # print("Y: ", Y)
    print("predictions:", predictions)


    acc = accuracy_score(predictions.detach(), Y)
    return acc


def initialise(dataset, args):
    """
    initialises GCN, optimiser, normalises graph, and features, and sets GPU number

    arguments:
    dataset: the entire dataset (with graph, features, labels as keys)
    args: arguments

    returns:
    a dictionary with model details (hypergcn, optimiser)
    """

    HyperGCN = {}
    V, E = dataset['n'], dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']

    # hypergcn and optimiser
    args.d, args.c = X.shape[1], Y.shape[1]
    hypergcn = networks_multilabels.HyperGCN(V, E, X, args)
    optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)

    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))

    # labels
    Y = np.array(Y)
    # Y = torch.LongTensor(np.where(Y)[1])
    Y = torch.FloatTensor(Y)

    # cuda
    args.Cuda = args.cuda and torch.cuda.is_available()
    if args.Cuda:
        hypergcn.cuda()
        X, Y = X.cuda(), Y.cuda()

    # update dataset with torch autograd variable
    dataset['features'] = Variable(X)
    dataset['labels'] = Variable(Y)

    # update model and optimiser
    HyperGCN['model'] = hypergcn
    HyperGCN['optimiser'] = optimiser
    # HyperGCN['total_loss'] = torch.tensor([0.0])
    return HyperGCN


def initialise_from_prev_model(prev_hypergcn_model, dataset, args):
    """
     initialises GCN, optimiser, normalises graph, and features, and sets GPU number

     arguments:
     dataset: the entire dataset (with graph, features, labels as keys)
     args: arguments

     returns:
     a dictionary with model details (hypergcn, optimiser)
     """

    HyperGCN = {}
    V, E = dataset['n'], dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']

    # hypergcn and optimiser (load state_dict of previous trained iteration)
    args.d, args.c = X.shape[1], Y.shape[1]
    hypergcn = networks_multilabels.HyperGCN(V, E, X, args)
    hypergcn.load_state_dict(prev_hypergcn_model['model'].state_dict())
    optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)
    optimiser.load_state_dict(prev_hypergcn_model['optimiser'].state_dict())


    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))

    # labels
    Y = np.array(Y)
    # Y = torch.LongTensor(np.where(Y)[1])
    Y = torch.FloatTensor(Y)

    # cuda
    args.Cuda = args.cuda and torch.cuda.is_available()
    if args.Cuda:
        hypergcn.cuda()
        X, Y = X.cuda(), Y.cuda()

    # update dataset with torch autograd variable
    dataset['features'] = Variable(X)
    dataset['labels'] = Variable(Y)

    # update model and optimiser
    HyperGCN['model'] = hypergcn
    HyperGCN['optimiser'] = optimiser
    # HyperGCN['total_loss'] = prev_hypergcn_model['total_loss']
    return HyperGCN


def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)
