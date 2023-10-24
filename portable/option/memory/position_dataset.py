import torch
import numpy as np 
import math 
import os 
import pickle 

from .set_dataset import SetDataset

def factored_minigrid_formatter(x):
    objects = ["agent",
               "key",
               "door",
               "goal",
               "box",
               "ball"]
    
    new_x = np.zeros((6, 3))
    
    for obj_idx, obj in enumerate(objects):
        new_x[obj_idx] = x[obj]
    
    return new_x

class PositionSet(SetDataset):
    def __init__(self,
                 data_formatter,
                 batchsize=16, 
                 max_size=100000):
        super().__init__(batchsize, max_size)

        self.data_formatter = data_formatter
        
        self.max = None
        self.min = None
    
    def _get_norm_stats(self):
        self.max = torch.zeros((1,1,3))
        self.min = torch.zeros((1,1,3))
        data = torch.from_numpy(np.array([])).float()
        if self.true_length > 0:
            data = self.concatenate(data, self.true_data)
        if self.false_length > 0:
            data = self.concatenate(data, self.false_data)
        if self.priority_false_length > 0:
            data = self.concatenate(data, self.priority_false_data)
        
        for idx in range(3):
            self.max[0,0,idx] = torch.max(data[:,:,idx])
            self.min[0,0,idx] = torch.min(data[:,:,idx])
        
    
    def transform(self, x):
        return (x-self.min)/(self.max-self.min)
    
    def add_true_files(self, file_list):
        for file in file_list:
            data = np.load(file, allow_pickle=True)
            file_data = []
            for x in data:
                point = self.data_formatter(x)
                file_data.append(point)
            file_data = torch.from_numpy(np.array(file_data)).float()
            self.true_data = self.concatenate(self.true_data, file_data)
        
        self.true_length = len(self.true_data)
        self._set_batch_num()
        self._get_norm_stats()
        self.shuffle()
        self.counter = 0
            
    def add_false_files(self, file_list):
        for file in file_list:
            data = np.load(file, allow_pickle=True)
            file_data = []
            for x in data:
                point = self.data_formatter(x)
                file_data.append(point)
            file_data = torch.from_numpy(np.array(file_data)).float()
            self.false_data = self.concatenate(self.false_data, file_data)
        
        self.false_length = len(self.false_data)
        self._set_batch_num()
        self._get_norm_stats()
        self.shuffle()
        self.counter = 0

    def add_priority_false_files(self, file_list):
        for file in file_list:
            data = np.load(file, allow_pickle=True)
            file_data = []
            for x in data:
                point = self.data_formatter(x)
                file_data.append(point)
            file_data = torch.from_numpy(np.array(file_data)).float()
            self.priority_false_data = self.concatenate(self.priority_false_data, file_data)
        
        self.priority_false_length = len(self.priority_false_data)
        self._set_batch_num()
        self._get_norm_stats()
        self.shuffle()
        self.counter = 0