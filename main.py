import argparse
import os
import torch
import numpy as np


import pandas as pd
# from meta_model import Meta_model
from mtask_model import Mtask_model

from dataset import  ECFPDataset, SmilesDataset
from torch.utils.data import DataLoader
import pandas as pd
from config import args




def main(split_idx):
    args.split_idx = split_idx

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # there are 78 protein in HME dataset
    train_percent = 80
    path_head = args.path_head

    args.dataset = "HME_dataset_{}".format(args.layer_size)
    print(args.dataset)

    
    # penalty coefficent for data balancing, num of inactive data / num of active data
    with open(path_head.split("split")[0] + "/train_{}_{}_penalty.txt".format(train_percent, split_idx), "r") as fw:
        penalty_coefficients = fw.readline().split()
        args.penalty_coefficients = list(map(float, penalty_coefficients))
    
    model = Mtask_model(args).to(device)

    valid_data_list = []
    valid_data_taskID = []
   
    # get data in each task
    s_train, l_train = [], [],
    weights = []
    for task_idx in range(args.num_tasks):
        task_id = args.task_no[task_idx]
        # trian set
        with open(path_head + str(task_id) + '/train_{}_{}.txt'.format(train_percent, split_idx), 'r') as fw:
            for line in fw.readlines():
                smiles, label = line.split()
                label = int(label)
                if smiles not in s_train:
                    s_train.append(smiles)

                    labels = [0] * args.num_tasks
                    labels = np.array(labels)
                    labels[task_idx] = label
                    l_train.append(labels)

                    weight = labels.copy()
                    weight[task_idx] = 1
                    weights.append(weight)
                else:
                    idx = s_train.index(smiles)
                    l_train[idx][task_idx] = label
                    weights[idx][task_idx] = 1

        # valid set
        with open(path_head + str(task_id) + '/valid_{}_{}.txt'.format(train_percent, split_idx), 'r') as fw:
            s_test, l_test = [], []
            for line in fw.readlines():
                smiles, label = line.split()
                label = int(label)
                s_test.append(smiles)
                l_test.append(label)

        task_valid_set =  ECFPDataset(s_test, l_test, task_id)
        # valid data
        valid_data = DataLoader(task_valid_set, batch_size = args.batch_size_eval)
        valid_data_list.append(valid_data)
        # valid ids are mapped with test_data_list
        valid_data_taskID.append(task_valid_set.get_task_id())


    train_set = DataLoader(SmilesDataset(s_train, l_train, weights), batch_size = args.batch_size)



    new_dataset_param_prefix = "params/"+ args.dataset + "_param/"
    if not os.path.exists(new_dataset_param_prefix ):
        os.makedirs(new_dataset_param_prefix)
    pretrain_model_path = new_dataset_param_prefix + 'parameter_train_{}_{}'.format(train_percent, split_idx)

    new_dataset_res_prefix = "result/" + args.dataset + "_res/"
    if not os.path.exists(new_dataset_res_prefix ):
        os.makedirs(new_dataset_res_prefix)

    result_path = new_dataset_res_prefix + args.dataset 
    best_accs = 0
    best_auc = 0
    best_aupr = 0
    best_f1 = 0
    best_ba = 0
   
    epoch = 0
    max_epoch = args.epoch
    
    while epoch < max_epoch + 1:
    # while False:
        torch.cuda.empty_cache()
        model.train(train_set)

        if epoch % 50 == 0:
            accs, auc, aupr, f1, ba = model.test_for_comp(valid_data_list, valid_data_taskID)
            
            if auc > best_auc:
                best_auc = auc
                best_aupr = aupr
                best_accs = accs
                best_f1 = f1
                best_ba = ba
                torch.save(model.state_dict(), pretrain_model_path+'_best_valid_auc.pkl')

            fw = open(result_path + "_train_{}_{}_valid.txt".format(train_percent, split_idx), "a+")
            
            fw.write("*"*20+"epoch: " + "\t")
            fw.write(str(epoch) + "\t")
            fw.write("*"*20 + "\n")

            fw.write("valid set: ACC = ")
            fw.write(str(accs))
            fw.write("\t AURoc = {} \t".format(auc))
            fw.write("AUPR = {} \t".format(aupr))
            fw.write("f1 = {} \t".format(f1))
            fw.write("ba = {} \t".format(ba))
            fw.write('\n')

            fw.write("best:\t ACC = ")
            fw.write(str(best_accs))
            fw.write("\t AURoc = {} \t".format(best_auc))
            fw.write("AUPR = {} \t".format(best_aupr))
            fw.write("f1 = {} \t".format(best_f1))
            fw.write("ba = {} \t".format(best_ba))
            fw.write("\n")
            fw.close()

        epoch += 1



if __name__ == "__main__":
    for split_idx in [0,1,2]:
        main(split_idx)
