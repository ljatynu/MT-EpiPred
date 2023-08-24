import pandas as pd
import numpy as np
import os
from tqdm import tqdm

path_head = "dataset/HME_dataset/"
task_split_path_head = path_head + "task_split/"
data_df = pd.read_csv(path_head + "datapoints.csv")
random_seeds = [0, 32, 1024]

protein_list = data_df.columns[2:].values.tolist()


# split data by task
if not os.path.exists(task_split_path_head):
    os.makedirs(task_split_path_head)

for task_idx, p in enumerate(tqdm(protein_list, desc="split data by task")):
    task_data_s = []
    task_data_label = []
    path = task_split_path_head + str(task_idx) + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    for cmpd_idx in range(len(data_df)):
        val = data_df.loc[cmpd_idx, p]
        if not np.isnan(val):
            s = data_df.loc[cmpd_idx, "Smiles"]
            task_data_s.append(s)
            task_data_label.append(int(val))
    
    task_data = {
                "Smiles" : task_data_s,
                "label" : task_data_label
    }
    task_data_df = pd.DataFrame(task_data)
    task_data_df.to_csv(path+"data.csv", index=None)


# split train valid test set
task_num = len(protein_list)
split_file_path_tail = "/data.csv"
from sklearn.model_selection import train_test_split

def write_train_test_set(data, split_file_path, random_seed, idx, train_percent = 0.8, label_type = None, task_id = None, write_type = "a+"):
    np.random.seed(random_seed)
    total_num = len(data)
    if total_num == 0:
        print("task {} has no {}!".format(task_id, label_type))
        return
        
    if total_num >= 10:
        c_train, c_valid = train_test_split(range(total_num), train_size=train_percent, random_state=random_seed)
        c_valid, c_test = train_test_split(c_valid, test_size=0.5, random_state=0)
    else:
        # 如果总数不够10个，但超过3个,那么随机取一个测试样本和一个验证样本
        if total_num >= 3:
            c_valid, c_test = np.array_split(np.random.choice(total_num, size=2, replace=False),2)
            c_train = list(range(total_num))
            for x in np.vstack((c_valid, c_test)):
                c_train.remove(x)
            # 随机打乱c_train
            np.random.shuffle(c_train)
        # 如果只有2个，取一个作为训练样本，测试样本和验证样本为同一样本
        else:
            c_valid, c_train = np.array_split(np.random.choice(total_num, size=2, replace=False),2)
            c_test = c_valid
        
    train_file = split_file_path + "/train_" + str(int(train_percent * 100)) + "_{}".format(idx) + ".txt"
    test_file = split_file_path + "/test_" + str(int(train_percent * 100)) + "_{}".format(idx) + ".txt"
    valid_file = split_file_path + "/valid_" + str(int(train_percent * 100)) + "_{}".format(idx) + ".txt"

    # write train set
    with open(train_file, write_type) as fw:
        for idx in c_train:
            smiles = data.iloc[idx]["Smiles"]
            label = data.iloc[idx]["label"]
            fw.write(smiles + " " + str(int(label)) + "\n")
    #write valid set
    with open(test_file, write_type) as fw:
        for idx in c_test:
            smiles = data.iloc[idx]["Smiles"]
            label = data.iloc[idx]["label"]
            fw.write(smiles + " " + str(int(label)) + "\n")
    #write test set
    with open(valid_file, write_type) as fw:
        for idx in c_valid:
            smiles = data.iloc[idx]["Smiles"]
            label = data.iloc[idx]["label"]
            fw.write(smiles + " " + str(int(label)) + "\n")

train_percent = 0.8
split_file_path_head = path_head + "/split/"
for task_id in tqdm(range(task_num), desc="train_valid_test split"):
    split_file_path= split_file_path_head + str(task_id)
    if not os.path.exists(split_file_path):
        os.makedirs(split_file_path)
    path = task_split_path_head + str(task_id) + split_file_path_tail
    data_d = pd.read_csv(path)
    data_pos = data_d[data_d["label"] == 1].reset_index(drop=True)
    data_neg = data_d[data_d["label"] == 0].reset_index(drop=True)
    
    for idx, random_seed in enumerate(random_seeds):
        write_train_test_set(data_pos, split_file_path, random_seed, idx, train_percent, label_type = "positive label", task_id = task_id, write_type="w+")
        write_train_test_set(data_neg, split_file_path, random_seed, idx, train_percent, label_type = "negtive label", task_id = task_id)


# get penalty for each task
task_ids = range(task_num)
train_percent = 80
for idx in [0,1,2]:
    penalty_path_tail = '/train_{}_{}.txt'.format(train_percent, idx)
    penalty = []
    dir_path = path_head
    file_name = dir_path + 'train_{}_{}_penalty.txt'.format(train_percent, idx)
    for i in tqdm(task_ids, desc="get penalty"):
        with open(split_file_path_head + str(i) + penalty_path_tail, 'r') as fw:
            neg_counter = 0.0
            pos_counter = 0.0
            for line in fw.readlines():
                if int(line.split()[1]) == 1:
                    pos_counter += 1
                else:
                    neg_counter += 1    
            penalty.append(neg_counter/pos_counter)

    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_name, 'w+') as fw2:
        for val in penalty:
            fw2.write(str(val) + " ")