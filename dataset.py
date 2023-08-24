"""
@author: Xie Xingran
@Date: 21-12-27
"""


from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
import torch
import random
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np


class SmilesDataset(Dataset):
    '''
    each compound has n labels,
    n mean the total number of task,
    b means the total number of compounds.
    '''
    def __init__(self, smiles, labels, weights):
        '''
        @Params:
            simels -> strings (1 x b)
            labels -> np.array (b x n)
            weight -> np.array (b x n) : 
                the index of corespoding task with actual datapoint(has label 1 or 0) will be set to 1, 
                else(has no label) will be set to 0
        '''
        self.smiles = smiles
        self.labels = labels
        self.weights = weights
    
    def __len__(self):
        '''
        get total number of compounds
        '''
        return len(self.smiles)

    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor(b x 1024)
            key: "labels", val: active(1) or inactive(0) -> tensor(b x n)} -> dict
            weights -> tensor(b x n)
        '''
        # print(item)
        s = self.smiles[idx]
        fp = self.process(s)
        output = {"vec": fp,
                  "labels": self.labels[idx]}
        weight = self.weights[idx]         

        return {key: torch.tensor(value) for key, value in output.items()}, weight
    
    def process(self, s):
        
        return self.getECFP(s)
        
    def getECFP(self, smiles):
        '''
        compute ECFP4 by rdkit
        @return: 
            ECFP4 -> np.array
        '''
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

class ECFPDataset(Dataset):
    def __init__(self, smiles, labels, task_id=-1):
        self.smiles = smiles
        self.labels = labels 
        # use to distinguish different task
        self.task_id = task_id

    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        '''
        @return:
            {key: "vec", val: ECPF4 -> torch.tensor
            key: "label", val: active(1) or inactive(0) -> tensor}
        '''
        # print(item)
        
        s = self.smiles[idx]
        fp = self.process(s)
        output = {"vec": fp,
                  "label": self.labels[idx]}         

        return {key: torch.tensor(value) for key, value in output.items()}

    # def load(self, index):
    #     self.ecfp = [self.getECFP(self.smiles[i]) for i in index]
    #     self.labels = [self.labels[i] for i in index]
    #     # self.corpus_lines = len(self.smiles)
    #     return self

    def process(self, s):
        
        return self.getECFP(s)
    
    def get_task_id(self):
        return self.task_id
        
    def getECFP(self, smiles):
        '''
        compute ECFP4 by rdkit
        @return: 
        ECFP4 -> np.array
        '''
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr
