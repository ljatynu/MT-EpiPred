from enum import EnumMeta
import sys
import os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import torch
import torch.nn as nn
from model import MT_DNN
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, \
                            balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score
import numpy as np


class Mtask_model(nn.Module):
    def __init__(self,args):
        super(Mtask_model, self).__init__()
    
        self.dataset = args.dataset
        self.emb_dim = args.emb_dim
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.decay = args.decay
        self.layer_size = args.layer_size
        self.split_idx = args.split_idx


        # self.criterion = nn.CrossEntropyLoss()
        self.task_no = args.task_no
        
        self.penalty_coefficients = args.penalty_coefficients

        self.mtask_model = MT_DNN(args.emb_dim, args.task_no, args.layer_size, args.num_tasks)

        model_param_group = []
        model_param_group.append({"params": self.mtask_model.parameters()})

        self.opt = optim.Adam(model_param_group, lr = self.lr, weight_decay=self.decay)
        # self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, input):
        '''
        return task_num predictions
        '''
        return self.mtask_model(input)
    

    def predict(self, init_feat):
        self.softmax = nn.Softmax(dim=1)
        tasks_preds = self.forward(init_feat)

        for taskID_idx, batch_preds in enumerate(tasks_preds):
            batch_preds = self.softmax(batch_preds)
            tasks_preds[taskID_idx] = batch_preds[:, 1]

        
        return tasks_preds




    def train(self, multi_task_train_data):
        
        self.mtask_model.train()

        for _, (batch, weights) in enumerate(tqdm(multi_task_train_data, desc="{}-{}".format(self.layer_size, self.split_idx))):
            weights = weights.to(self.device)
            init_features = batch["vec"].to(self.device)
           
            # Get the prediction of the each task
            # Since nn.CrossEntropyLoss include softmax function, there is not softmax in preds computation here.
            preds = self.forward(init_features)
            preds = torch.stack(preds).transpose(0,1).to(self.device)

            y = batch['labels'].to(self.device).to(torch.long)
            criterion = nn.CrossEntropyLoss(reduction='none')
           
            # now there are n calss losses
            # loss -> [b, n]
            losses = []
            for idx, pred in enumerate(preds):
                loss = criterion(pred.double(), y[idx])
                # set the loss to 0 where the task has no datapoint
                # just loss x weights will achieve this point
                loss = loss.mul(weights[idx])
                losses.append(loss)
           
            losses = torch.stack(losses).to(self.device)
            
            # change the penalty according to true label, (penalty when y = 1, no penalty(1) when y = 0)
            penalty = torch.DoubleTensor(self.penalty_coefficients).to(self.device).repeat(y.size()[0], 1).mul(y.to(torch.float))
            one = torch.ones_like(penalty)
            penalty = torch.where(penalty == 0, one, penalty)            

            # add(mul) penalty to losses
            losses = losses.mul(penalty)

            self.opt.zero_grad()
            losses.backward(losses.clone().detach())

           
            self.opt.step()
    
    def test(self, test_data_list, test_data_taskID):
        '''
        return auc, acc, aupr, f1 list
        '''
        aucs = []
        accs = []
        auprs = []
        f1s = []
        bas = []
        mccs = []
        precisions = []
        recalls = []
        self.mtask_model.eval()
        
        for taskID_idx, test_data in enumerate(test_data_list):
            y_true = []
            y_scores = []
            task_id = test_data_taskID[taskID_idx]
            print('task_id: ', task_id)
            for i, batch in enumerate(tqdm(test_data, desc="Iteration")):
                init_features = batch["vec"].to(self.device)
                # Get the prediction of the corresponding task
                preds = self.predict(init_features)[taskID_idx]
                
                y_scores.append(preds)
                y_true.append(batch['label'].to(self.device).view(preds.shape))

            y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
            y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
            y_preds = y_scores.copy()
            y_preds[y_preds < 0.5] = 0
            y_preds[y_preds != 0] = 1 

            
            
            auc = roc_auc_score(y_true, y_scores)
            acc = accuracy_score(y_true, y_preds)
            aupr = average_precision_score(y_true, y_scores)
            f1 = f1_score(y_true, y_preds)
            ba = balanced_accuracy_score(y_true, y_preds)
            mcc = matthews_corrcoef(y_true, y_preds)
            precisions.append(precision_score(y_true, y_preds))
            recalls.append(recall_score(y_true, y_preds))
            
            aucs.append(auc)
            accs.append(acc)
            auprs.append(aupr)
            f1s.append(f1)
            bas.append(ba)
            mccs.append(mcc)
            
        return aucs, accs, auprs, f1s, bas, mccs, precisions, recalls

    def test_for_comp(self, test_data_list, test_data_taskID):
        '''
        return a average auc
        '''
        self.mtask_model.eval()
        y_true = []
        y_scores = []
        for taskID_idx, test_data in enumerate(test_data_list):
            
            task_id = test_data_taskID[taskID_idx]
            print('task_id: ', task_id)
            for i, batch in enumerate(tqdm(test_data, desc="Iteration")):
                init_features = batch["vec"].to(self.device)
                # Get the prediction of the corresponding task
                preds = self.predict(init_features)[taskID_idx]
                
                y_scores.append(preds)
                y_true.append(batch['label'].to(self.device).view(preds.shape))

        y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
        y_predict = np.squeeze(y_scores.copy())
        for idx, val in enumerate(y_predict):
            if val < 0.5 :
                y_predict[idx] = 0
            else:
                y_predict[idx] = 1
                

        
        auc = roc_auc_score(y_true, y_scores)
        acc = accuracy_score(np.squeeze(y_true), y_predict)
        aupr = average_precision_score(np.squeeze(y_true), y_scores)
        f1 = f1_score(np.squeeze(y_true), y_predict)
        ba = balanced_accuracy_score(y_true, y_predict)
        return acc, auc, aupr, f1, ba

    
        








        

        


        