import torch 
import numpy as np 
from experiments.encoder.core.models import *
from portable.option.memory import SetDataset

class Classifier():
    def __init__(self,
                 model_type,
                 input_size,
                 lr,
                 use_gpu,
                 embedding):
        assert model_type in MODEL_TYPE
        if model_type == MODEL_TYPE[0]:
            self.model = SimpleClassifier(input_size=input_size)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.use_gpu = use_gpu
        self.embedding = embedding
        if use_gpu:
            self.model.to("cuda")
            self.embedding.to("cuda")
        
        self.dataset = SetDataset(max_size=1000000,
                                  batchsize=32,
                                  create_validation_set=True)
        
    def add_data(self,
                 positive_files,
                 negative_files):
        assert isinstance(positive_files, list)
        assert isinstance(negative_files, list)
        
        if len(positive_files) > 0:
            self.dataset.add_true_files(positive_files)
        
        if len(negative_files) > 0:
            self.dataset.add_false_files(negative_files)
    
    def train(self,
              epochs):
        
        self.model.train()
        loss_tracker = []
        loss_tracker_val = []
        accuracy_tracker = []
        accuracy_tracker_val = []
        for epoch in range(epochs):
            self.dataset.shuffle()
            losses = 0
            losses_val = 0
            accuracies = 0
            accuracies_val = 0
            counter = 0
            for _ in range(self.dataset.num_batches):
                counter += 1
                x, y, x_val, y_val = self.dataset.get_batch()
                if self.use_gpu:
                    x = x.to("cuda")
                    y = y.to("cuda")
                
                x = self.embedding.feature_extractor(x)
                pred_y = self.model(x)
                loss = self.criterion(pred_y, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                losses += loss.item()
                pred_class = torch.argmax(pred_y, dim=1).detach()
                accuracies += torch.sum(pred_class == y)/len(y)
            
                with torch.no_grad():
                    if self.use_gpu:
                        x_val = x_val.to("cuda")
                        y_val = y_val.to("cuda")
                    
                    x_val = self.embedding.feature_extractor(x_val)
                    pred_y_val = self.model(x_val)
                    loss_val = self.criterion(pred_y_val, y_val)
                    losses_val += loss_val
                    pred_class_val = torch.argmax(pred_y_val, dim=1).detach()
                    accuracies_val += torch.sum(pred_class_val == y_val)/len(y_val)
                    
            
            print("Epoch {}".format(epoch))
            print("classifier loss: {}".format(losses/counter))
            print("classifier acc: {}".format(accuracies/counter))
            print("classifier loss val: {}".format(losses_val/counter))
            print("classifier acc val: {}".format(accuracies_val/counter))
            print("============================================================")

            loss_tracker.append(losses/counter)
            loss_tracker_val.append(losses_val/counter)
            accuracy_tracker.append(accuracies/counter)
            accuracy_tracker_val.append(accuracies_val/counter)
        
        return loss_tracker, accuracy_tracker, loss_tracker_val, accuracy_tracker_val
    
    
