import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import dropout



class MT_DNN(torch.nn.Module):
    def __init__(self, in_fdim, task_ids, layer_size, num_tasks):
        super(MT_DNN, self).__init__()
        self.in_fdim = in_fdim
        # self.out_fdim = out_fdim
        self.layer_size = layer_size
        self.num_tasks = num_tasks

        self.num_tasks = len(task_ids)
        # task id maps head No.
        self.tasks_id_maps_heads = {}
        for idx,task_id in enumerate(task_ids):
            self.tasks_id_maps_heads[task_id] = idx

        last_fdim = self.create_bond()
        self.create_sphead(last_fdim)
        
        

    def create_bond(self):
        '''
        Creates the feed-forward layers for the model.
        '''
        activation = nn.ReLU()
        dropout = nn.Dropout(p=0.5)     
        last_fdim = 1000

        ffn = []

        # Create FFN layers
        if self.layer_size == 'shallow':
            # [1000]
            ffn.extend([
                torch.nn.Linear(self.in_fdim, 1000),
                activation,
            ])
        
        elif self.layer_size == 'moderate':
            # [1500, 1000]
            ffn.extend([
                torch.nn.Linear(self.in_fdim, 1500),
                activation
            ])
            ffn.extend([
                dropout,
                torch.nn.Linear(1500, 1000),
                activation
            ])
            #last_fdim = 1000
        
        elif self.layer_size == 'deep':
            # [2000, 1000, 500]
            ffn = [
                torch.nn.Linear(self.in_fdim, 2000),
                activation
            ]
            for i in range(1,3):
                ffn.extend([
                    dropout,
                    torch.nn.Linear(2000//i, 1000//i),
                    activation
                ])
            last_fdim = 500
        elif self.layer_size == 'task_relate':
            # [1024, num_tasks]
            ffn.extend([
                torch.nn.Linear(self.in_fdim, 1024),
                activation
            ])
            ffn.extend([
                dropout,
                torch.nn.Linear(1024, self.num_tasks),
                activation
            ])
            last_fdim = self.num_tasks
        
        else:
             raise ValueError("unmatched layer_size(shallow, moderate, deep, p_best).")
            
           
        self.bond = torch.nn.Sequential(*ffn)
        
        return last_fdim
    
    def create_sphead(self,last_fdim):
        '''
        create task specific output layers
        '''
        heads = []
        if self.num_tasks < 1:
            raise ValueError("unmatched task_num, which must greater than 1.")
        
        # each task has it's own specific output layer(head)
        for _ in range(self.num_tasks):
            #define the task as a bio-classifing problem
            ffn = nn.Linear(last_fdim, 2)
            heads.append(ffn)
        
        self.heads = nn.ModuleList(heads)
    
    def forward(self, in_feat):
        out_feat = self.bond(in_feat)
   
        # compute output for each task through corresponding head
        output = []
        for head in self.heads:
            output.append(head(out_feat))
        
        return output

    