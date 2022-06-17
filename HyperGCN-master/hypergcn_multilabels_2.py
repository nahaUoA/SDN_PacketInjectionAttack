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
from data import data_multilabels

# load train data
train_datapath = os.path.join(Dir, "data", args.data, "train")
train_dataset_dirs = [o for o in os.listdir(train_datapath) if os.path.isdir(os.path.join(train_datapath, o))]
train_datasets = [None for i in range(len(train_dataset_dirs))]
trains = [None for i in range(len(train_dataset_dirs))]
for i in range(len(train_dataset_dirs)):
    data_arg = {'data': args.data, 'dataset': train_dataset_dirs[i], 'usage': 'train'}
    train_datasets[i], trains[i] = data_multilabels.load_data(data_arg)

# # load test data
test_datapath = os.path.join(Dir, "data", args.data, "test")
test_dataset_dirs = [o for o in os.listdir(test_datapath) if os.path.isdir(os.path.join(test_datapath, o))]
test_datasets = [None for i in range(len(test_dataset_dirs))]
tests = [None for i in range(len(test_dataset_dirs))]
for i in range(len(test_dataset_dirs)):
    data_arg = {'data': args.data, 'dataset': test_dataset_dirs[i], 'usage': 'test'}
    test_datasets[i], tests[i] = data_multilabels.load_data(data_arg)


from model import model_multilabels_2

#train HyperGCN
print("training ...")
HyperGCN = model_multilabels_2.sequential_train(train_datasets, trains, args)






# from model import model_multilabels
#HyperGCN = model_multilabels.initialise(train_datasets[0], args)

# for i in range(len(train_datasets)):
#     print(train_dataset_dirs[i])
#     if i == 0:
#         HyperGCN = model_multilabels.train(HyperGCN, train_datasets[i], trains[i], args)
#         continue
#     # HyperGCN = model_multilabels.initialise_from_prev_model(HyperGCN, train_datasets[i], args)
#     HyperGCN = model_multilabels.initialise(train_datasets[i], args)
#     HyperGCN = model_multilabels.train(HyperGCN, train_datasets[i], trains[i], args)
#     # trace print for the learnable parameters
#     print(str(i) + "th iteration model's after trained state_dict:")
#     for param_tensor in HyperGCN['model'].state_dict():
#         print(param_tensor, "\t", HyperGCN['model'].state_dict()[param_tensor].shape)
#         print(param_tensor, "\t", HyperGCN['model'].state_dict()[param_tensor])





# test hyperGCN
# innit again for the testing network, using parameter learned from the train model above
model_acc = []
print("testing...")
for i in range(len(test_datasets)):
    print(test_dataset_dirs[i])
    HyperGCN = model_multilabels_2.initialise_from_prev_model(HyperGCN, test_datasets[i], args)
    acc = model_multilabels_2.test(HyperGCN, test_datasets[i], tests[i], args)
    print("accuracy:", float(acc), ", error:", float(1-acc))
    model_acc.append(acc)
    if acc != 1 and acc != 0:
        print("found")
        break

avg_acc = sum(model_acc)/len(test_datasets)
print("average accuracy across all test data: ", float(avg_acc))

