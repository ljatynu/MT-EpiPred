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

from my_utils import record_metrics_per_task,get_res_per_task
from config import args


def main(layer_size):
    # Training settings
    # w_test and w_valid are used to compute weighted mean 
    weight_per_task_df = pd.read_csv(args.path_head.split("split")[0] + "/active_inactive_statistics.csv")
    c = weight_per_task_df["test_neg"].values + weight_per_task_df["test_pos"].values
    w_test = c/np.sum(c)
    c_v = weight_per_task_df["valid_neg"].values + weight_per_task_df["valid_pos"].values
    w_valid = c_v/np.sum(c_v)


    for split_idx in [0, 1, 2]:
        args.split_idx = split_idx
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.runseed)

        # there are 78 protein in HDM dataset
        args.task_no = range(78)
        args.num_tasks = len(args.task_no)
        train_percent = 80
        path_head = args.path_head
        args.penalty_coefficients = None

        args.dataset = "HME_dataset"
        print(args.dataset)

        test_data_list = []
        test_data_taskID = []
        valid_data_list = []
        valid_data_taskID = []

    
        # get data in each task
        for task_idx in range(args.num_tasks):
            task_id = args.task_no[task_idx]

            # test set
            with open(path_head + str(task_id) + '/test_{}_{}.txt'.format(train_percent, split_idx), 'r') as fw:
                s_test, l_test = [], []
                for line in fw.readlines():
                    smiles, label = line.split()
                    label = int(label)
                    s_test.append(smiles)
                    l_test.append(label)

            task_test_set =  ECFPDataset(s_test, l_test, task_id)
            # test data
            test_data = DataLoader(task_test_set, batch_size = args.batch_size_eval)
            test_data_list.append(test_data)
            # task ids are mapped with test_data_list
            test_data_taskID.append(task_test_set.get_task_id())

            # valid set
            with open(path_head + str(task_id) + '/valid_{}_{}.txt'.format(train_percent, split_idx), 'r') as fw:
                s_valid, l_valid = [], []
                for line in fw.readlines():
                    smiles, label = line.split()
                    label = int(label)
                    s_valid.append(smiles)
                    l_valid.append(label)

            task_valid_set =  ECFPDataset(s_valid, l_valid, task_id)
            # valid data
            valid_data = DataLoader(task_valid_set, batch_size = args.batch_size_eval)
            valid_data_list.append(valid_data)
            # task ids are mapped with valid_data_list
            valid_data_taskID.append(task_valid_set.get_task_id())


        args.layer_size = layer_size
        model = Mtask_model(args).to(device)

        new_dataset_param_prefix = "params/HME_dataset_{}_param/".format(layer_size)
        
        pretrain_model_path = new_dataset_param_prefix + 'parameter_train_{}_{}'.format(train_percent, split_idx)

        new_dataset_res_prefix = "result/{}_{}_res/".format(args.dataset, layer_size)
        result_path = new_dataset_res_prefix  + "/{}_test_valid_per_task/".format(layer_size)
        if not os.path.exists(result_path ):
            os.makedirs(result_path)
        
        # load parameter
        model.load_state_dict(torch.load(pretrain_model_path+'_best_valid_auc.pkl'))

        
        # test
        get_res_per_task(model, test_data_list, test_data_taskID, args.dataset, result_path + "test_{}".format(split_idx), w_test)
        # valid
        get_res_per_task(model, valid_data_list, valid_data_taskID, args.dataset, result_path + "valid_{}".format(split_idx), w_valid)

    






    

if __name__ == "__main__":
    # dataset
    main("shallow")
