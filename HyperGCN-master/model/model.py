from model import networks
from sklearn.metrics import f1_score
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F

from torch.autograd import Variable
from tqdm import tqdm
from model import utils
from sklearn.metrics import precision_recall_fscore_support as score



def train(HyperGCN, dataset, T, args):
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
    # weight = torch.tensor([1.0, 12.4])
    # print(weight)


    for epoch in tqdm(range(args.epochs)):

        optimiser.zero_grad()
        Z = hypergcn(X)
        # print(Z.shape)
        loss = F.cross_entropy(Z[T], Y[T])
        print(loss)

        loss.backward()
        optimiser.step()

    HyperGCN['model'] = hypergcn
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
    acc = accuracy(Z[t], Y[t])
    precision, recall, fscore, support, f1 = metrics(Z[t], Y[t])
    return acc, precision, recall, fscore, support, f1


def metrics(Z, Y):
    predictions = Z.max(1)[1].type_as(Y)
    precision, recall, fscore, support = score(Y.cpu().numpy(), predictions.cpu().numpy())
    f1 = f1_score(Y.cpu().numpy(), predictions.cpu().numpy(), average='macro')
    return precision, recall, fscore, support, f1




def accuracy(Z, Y):
    """
    arguments:
    Z: predictions
    Y: ground truth labels

    returns: 
    accuracy
    """
    
    predictions = Z.max(1)[1].type_as(Y)
    torch.set_printoptions(threshold=np.inf)

    # print("Z: ", Z.shape, Z)
    # print("Y: ", Y.shape, Y)
    # atker = 0
    # for i, label in enumerate(Y):
    #     if label == 1:
    #         atker += 1
    #         print(i, label)
    # print("Number of attacker: ", atker)

    # print("pred:", predictions)
    # atker_pred_count = 0
    # atker_correct_count = 0
    # for i, pred in enumerate(predictions):
    #     if pred == 1:
    #         print(i, pred)
    #         atker_pred_count += 1
    #         if Y[i] == 1:
    #             atker_correct_count += 1
    correct = predictions.eq(Y).double()
    correct = correct.sum()
    # print("Number of attacker predicted: ", atker_pred_count)
    #print("Number of attacker correctly predicted: ", atker_correct_count)

    accuracy = correct / len(Y)
    return accuracy



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
    hypergcn = networks.HyperGCN(V, E, X, args)
    optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)


    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

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
    hypergcn = networks.HyperGCN(V, E, X, args)
    hypergcn.load_state_dict(prev_hypergcn_model['model'].state_dict())
    optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.rate, weight_decay=args.decay)
    optimiser.load_state_dict(prev_hypergcn_model['optimiser'].state_dict())

    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))

    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

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
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)
