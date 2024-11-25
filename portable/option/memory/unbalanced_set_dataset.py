import torch 
import numpy as np 
import math 
import os 
import pickle 
import gin

@gin.configurable
class UnbalancedSetDataset():
    def __init__(self,
                 batchsize=64,
                 unlabelled_batchsize=None,
                 max_size=100000,
                 data_dir=".",
                 class_weights=[0.5,0.5]
                 ):
        self.batchsize = batchsize
        if unlabelled_batchsize is not None:
            self.dynamic_unlabelled_batchsize = False
            self.unlabelled_batchsize = unlabelled_batchsize
        else:
            self.dynamic_unlabelled_batchsize = True
            self.unlabelled_batchsize = 0
        self.max_size = max_size
        
        self.data = torch.from_numpy(np.array([]))
        self.unlabelled_data = torch.from_numpy(np.array([]))
        self.labels = torch.from_numpy(np.array([]))
        
        self.data_length = 0
        self.unlabelled_data_length = 0
        self.counter = 0
        self.num_batches = 0
        self.unlabelled_counter = 0
        
        self.shuffled_indices = None
        self.shuffled_indices_unlabelled = None
        self.data_dir = data_dir
        self.class_weight = class_weights
    
    @staticmethod
    def transform(x):
        if torch.max(x) > 1:
            return (x/255.0).float()
        else:
            return x
    
    def get_equal_class_weight(self):
        num_positive = torch.sum(self.labels)
        num_negative = self.data_length - num_positive
        
        weighting = [
            self.class_weight[0]*2/num_negative+1e-5,
            self.class_weight[1]*2/num_positive+1e-5
        ]
        
        return weighting
    
    def reset(self):
        self.data = torch.from_numpy(np.array([]))
        self.labels = torch.from_numpy(np.array([]))
        self.counter = 0
        self.num_batches = 0
        self.unlabelled_counter = 0
        self.shuffled_indices = None
    
    def reset_memory(self):
        self.data = torch.from_numpy(np.array([]))
        self.labels = torch.from_numpy(np.array([]))
    
    def set_transform_function(self, transform):
        self.transform = transform
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(self.data, f)
        
        with open(os.path.join(path, "label.pkl"), "wb") as f:
            pickle.dump(self.labels, f)
    
    def load(self, path):
        data_file = os.path.join(path, "data.pkl")
        label_file = os.path.join(path, "label.pkl")
        if not os.path.exists(data_file):
            print("[UnbalancedSetDataset] No data found.")
            return
        if not os.path.exists(label_file):
            print("[UnbalancedSetDataset] No labels found.")
            return 
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)
        self.data_length = len(self.data)
        with open(label_file, "rb") as f:
            self.labels = pickle.load(f)
        
        self._set_batch_num()
        self.shuffle()
    
    def shuffle(self):
        self.shuffled_indices = np.random.permutation(self.data_length)
        self.shuffled_indices_unlabelled = np.random.permutation(range(self.unlabelled_data_length))
    
    def _set_batch_num(self):
        self.num_batches = math.ceil(
            self.data_length/self.batchsize
        )
        if self.dynamic_unlabelled_batchsize is True:
            self.unlabelled_batchsize = self.unlabelled_data_length//self.num_batches
    
    def add_true_files(self, file_list):
        for file in file_list:
            file = os.path.join(self.data_dir, file)
            data = np.load(file, allow_pickle=True)
            data = torch.from_numpy(data)
            if torch.max(data) <= 1:
                data = data*255
            data = data.int()
            data = data.squeeze()
            labels = torch.ones(len(data), dtype=torch.int8)
            self.data = self.concatenate(self.data, data)
            self.labels = self.concatenate(self.labels, labels)
            assert len(self.data) == len(self.labels)
            self.data_length = len(self.data)
            self._set_batch_num()
            self.counter = 0
            
            self.shuffle()
    
    def add_false_files(self, file_list):
        for file in file_list:
            file = os.path.join(self.data_dir, file)
            data = np.load(file, allow_pickle=True)
            data = torch.from_numpy(data)
            if torch.max(data) <= 1:
                data = data*255
            data = data.int()
            data = data.squeeze()
            labels = torch.zeros(len(data), dtype=torch.int8)
            self.data = self.concatenate(self.data, data)
            self.labels = self.concatenate(self.labels, labels)
            assert len(self.data) == len(self.labels)
            self.data_length = len(self.data)
            self._set_batch_num()
            self.counter = 0
            
            self.shuffle()
    
    def add_unlabelled_files(self, file_list):
        for file in file_list:
            file = os.path.join(self.data_dir, file)
            data = np.load(file)
            data = torch.from_numpy(data)
            if torch.max(data) <= 1:
                data = data*255
            data = data.int()
            data = data.squeeze()
            self.unlabelled_data = self.concatenate(self.unlabelled_data,
                                                    data)
        self.unlabelled_data_length = len(self.unlabelled_data)
        self._set_batch_num()
        self.unlabelled_counter = 0
        self.shuffle()
    
    def get_batch(self, shuffle_batch=True):
        data = []
        labels = []
        if (self.index() + self.batchsize) > self.data_length:
            data = self.data[self.shuffled_indices[self.index():]]
            labels = self.labels[self.shuffled_indices[self.index():]]
        else:
            data = self.data[self.shuffled_indices[self.index():self.index() + self.batchsize]]
            labels = self.labels[self.shuffled_indices[self.index():self.index() + self.batchsize]]
        
        data = self.transform(data)
        labels = labels.long()
        
        self.counter += 1

        print(f"Data shape: {data.shape}")
        return data, labels
    
    def get_unlabelled_batch(self):
        data = []
        index = self.unlabelled_index()
        if (index+self.unlabelled_batchsize) > len(self.unlabelled_data):
            num_remaining = self.unlabelled_data_length - index
            data = self.unlabelled_data[self.shuffled_indices_unlabelled[index:]]
            data = self.concatenate(data,
                                    data[self.shuffled_indices_unlabelled[:num_remaining]])
        else:
            data = self.unlabelled_data[self.shuffled_indices_unlabelled[index:index+self.unlabelled_batchsize]]
        
        data = self.transform(data)
        
        return data
    
    def index(self):
        return (self.counter*self.batchsize)%self.data_length
    
    def unlabelled_index(self):
        return (self.unlabelled_counter*self.unlabelled_batchsize)%self.unlabelled_data_length
    
    @staticmethod
    def concatenate(arr1, arr2):

        if len(arr1) == 0:
            return arr2
        else:
            return torch.cat((arr1, arr2), axis=0)