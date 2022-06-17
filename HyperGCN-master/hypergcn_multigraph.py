# parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))
from config import config
args = config.parse()



# seed
import os, inspect, torch, numpy as np
torch.manual_seed(args.seed)
np.random.seed(args.seed)



# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)


current = os.path.abspath(inspect.getfile(inspect.currentframe()))
Dir, _ = os.path.split(current)
from data import data_multigraph

# load train data
train_datapath = os.path.join(Dir, "data", args.data, "train")
train_dataset_dirs = [o for o in os.listdir(train_datapath) if os.path.isdir(os.path.join(train_datapath, o))]
train_datasets = [None for i in range(len(train_dataset_dirs))]
trains = [None for i in range(len(train_dataset_dirs))]
for i in range(len(train_dataset_dirs)):
    data_arg = {'data': args.data, 'dataset': train_dataset_dirs[i], 'usage': 'train'}
    print(train_dataset_dirs[i])
    train_datasets[i], trains[i] = data_multigraph.load_data(data_arg)

# # load test data
test_datapath = os.path.join(Dir, "data", args.data, "test")
test_dataset_dirs = [o for o in os.listdir(test_datapath) if os.path.isdir(os.path.join(test_datapath, o))]
test_datasets = [None for i in range(len(test_dataset_dirs))]
tests = [None for i in range(len(test_dataset_dirs))]
for i in range(len(test_dataset_dirs)):
    data_arg = {'data': args.data, 'dataset': test_dataset_dirs[i], 'usage': 'test'}
    test_datasets[i], tests[i] = data_multigraph.load_data(data_arg)


# # initialise HyperGCN
from model import model_multigraph


HyperGCN = model_multigraph.initialise(train_datasets[0], args)


#train HyperGCN
for i in range(len(train_datasets)):
    print(train_dataset_dirs[i])
    np.set_printoptions(threshold=10000)
    print(train_datasets[i]['labels'])
    if i == 0:
        HyperGCN = model_multigraph.train(HyperGCN, train_datasets[i], trains[i], args)
        continue

    HyperGCN = model_multigraph.initialise_from_prev_model(HyperGCN, train_datasets[i], args)

    # HyperGCN = model_multigraph.initialise(train_datasets[i], args)
    HyperGCN = model_multigraph.train(HyperGCN, train_datasets[i], trains[i], args)
    # trace print for the learnable parameters
    print(str(i) + "th iteration model's after trained state_dict:")
    for param_tensor in HyperGCN['model'].state_dict():
        print(param_tensor, "\t", HyperGCN['model'].state_dict()[param_tensor])




# test hyperGCN
# innit again for the testing network, using parameter learned from the train model above
model_acc = []
for i in range(len(test_datasets)):
    print(test_dataset_dirs[i])
    HyperGCN = model_multigraph.initialise_from_prev_model(HyperGCN, test_datasets[i], args)
    acc = model_multigraph.test(HyperGCN, test_datasets[i], tests[i], args)
    print("accuracy:", float(acc), ", error:", float(1-acc))
    model_acc.append(acc)

avg_acc = sum(model_acc)/len(test_datasets)
print("average accuracy across all test data: ", float(avg_acc))

