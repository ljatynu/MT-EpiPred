import numpy as np
import pandas as pd
from config import args

def record_metrics_per_task(result_path, res, metric_name, test_data_taskID, id2protein, weight):
        '''
        record auc for each task
        '''
        result_dict = {}

        fw = open(result_path + "_{}.txt".format(metric_name), "w+")
                
        fw.write("test: " + "\t")
        for idx, i in enumerate(res):
            protein_id = test_data_taskID[idx]
            protein_name = id2protein[protein_id]
            fw.write(protein_name + ":" + str(i) + "\t")
            result_dict[protein_name] = i
        fw.write("\n")

        fw.write("average test: "+ "\t")
        fw.write(str( np.average(res, weights=weight)) + "\t")
        result_dict["average"] = np.average(res, weights=weight)
        fw.write("\n")

        fw.write("\n")
        fw.close()

        result_df = pd.DataFrame.from_dict(result_dict, orient='index',columns=[metric_name]).reset_index().rename(columns={'index':'Name'})
        result_df.to_csv(result_path + "_{}.csv".format(metric_name), index=None)

def get_res_per_task(model, data_list, data_taskID, dataset, result_path, weight):
      # test
    aucs, accs, auprs, f1s, bas, mccs, precisions, recalls = model.test(data_list, data_taskID)
    protein_df = pd.read_csv(args.path_head.split("split")[0] + "/protein_list.csv")
    
    id2protein = protein_df.set_index("id")["Name"].to_dict()

    metrics_res = [aucs, accs, auprs, f1s, bas, mccs, precisions, recalls]
    for idx, metric in enumerate(["auc", "acc", "aupr", "f1", "BA", "MCC", "Precision", "Recall"]):
        record_metrics_per_task(result_path, metrics_res[idx], metric, data_taskID, id2protein, weight)

def get_group_ids(group_name):

    group_id_df = pd.read_csv("new_dataset/HME_dataset/group_id.csv")
    task_no = group_id_df[group_id_df[group_name].notna()][group_name].astype(int).values.tolist()

    return task_no