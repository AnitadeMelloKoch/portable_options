import logging 
import torch
import torch.nn as nn
import gin
import os
import numpy as np 

from portable.option.memory import SetDataset
from portable.option.divdis.models.small_cnn import SmallCNN
from portable.option.divdis.models.mlp import MultiHeadMLP, OneHeadMLP
from portable.option.divdis.divdis import DivDisLoss

logger = logging.getLogger(__name__)

MODEL_TYPE = [
    "one_head_mlp",
    "multi_head_mlp",
    "small_cnn"
]


@gin.configurable
class DivDisClassifier():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 
                 head_num,
                 learning_rate,
                 input_dim,
                 num_classes,
                 diversity_weight,
                 l2_reg_weight=0,
                 
                 dataset_max_size=1e6,
                 dataset_batchsize=32,
                 unlabelled_dataset_batchsize=None,
                 
                 summary_writer=None,
                 model_name='classifier') -> None:
        
        self.use_gpu = use_gpu,
        self.dataset = SetDataset(max_size=dataset_max_size,
                                  batchsize=dataset_batchsize,
                                  unlabelled_batchsize=unlabelled_dataset_batchsize)
        self.learning_rate = learning_rate
        
        self.head_num = head_num
        
        self.log_dir = log_dir
        
        self.classifier = SmallCNN(num_input_channels=input_dim,
                                   num_classes=num_classes,
                                   num_heads=head_num)
        #self.classifier = torch.compile(SmallCNN(num_input_channels=input_dim,
        #                                    num_classes=num_classes,
        #                                    num_heads=head_num),
        #                                #backend='tensorrt',
        #                                #dynamic=False
        #                                )
        
        
        self.optimizer = torch.optim.Adam(self.classifier.parameters(),
                                          lr=learning_rate,
                                          weight_decay=l2_reg_weight
                                          )
        
        self.divdis_criterion = DivDisLoss(heads=head_num)
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.diversity_weight = diversity_weight
    
    def save(self, path):
        torch.save(self.classifier.state_dict(), os.path.join(path, 'classifier_ensemble.ckpt'))
        self.dataset.save(path)
    
    def load(self, path):
        if os.path.exists(os.path.join(path, 'classifier_ensemble.ckpt')):
            print("classifier loaded from: {}".format(path))
            self.classifier.load_state_dict(torch.load(os.path.join(path, 'classifier_ensemble.ckpt')))
            self.dataset.load(path)
    
    def move_to_gpu(self):
        if self.use_gpu:
            self.classifier.to("cuda")
    
    def move_to_cpu(self):
        self.classifier.to("cpu")
    
    def add_data(self,
                 positive_files,
                 negative_files,
                 unlabelled_files):
        assert isinstance(positive_files, list)
        assert isinstance(negative_files, list)
        
        self.dataset.add_true_files(positive_files)
        self.dataset.add_false_files(negative_files)
        self.dataset.add_unlabelled_files(unlabelled_files)
    
    def train(self,
              epochs,
              start_offset=0):
        self.move_to_gpu()
        self.classifier.train()
        for epoch in range(start_offset, start_offset+epochs):
            self.dataset.shuffle()
            counter = 0
            
            class_loss_tracker = np.zeros(self.head_num)
            class_acc_tracker = np.zeros(self.head_num)
            div_loss_tracker = 0
            total_loss_tracker = 0
            
            self.dataset.shuffle()
            
            for _ in range(self.dataset.num_batches):
                counter += 1
                x, y = self.dataset.get_batch()
                unlabelled_x = self.dataset.get_unlabelled_batch()
                
                if self.use_gpu:
                    x = x.to("cuda")
                    y = y.to("cuda")
                    unlabelled_x = unlabelled_x.to("cuda")
                
                unlabelled_pred = self.classifier(unlabelled_x)
                pred_y = self.classifier(x)
                labelled_loss = 0
                for idx in range(self.head_num):
                    class_loss = self.ce_criterion(pred_y[ :,idx,:], y)
                    class_loss_tracker[idx] += class_loss.item()
                    pred_class = torch.argmax(pred_y[:,idx,:],dim=1).detach()
                    class_acc_tracker[idx] += (torch.sum(pred_class==y).item())/len(y)
                    labelled_loss += class_loss
                
                labelled_loss /= self.head_num
                
                div_loss = self.divdis_criterion(unlabelled_pred)
                
                div_loss_tracker += div_loss.item()
                
                objective = labelled_loss + self.diversity_weight*div_loss
                total_loss_tracker += objective.item()
                
                objective.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            logger.info("Epoch {}".format(epoch))
            for idx in range(self.head_num):
                logger.info("head {:.4f}: labelled loss = {:.4f} labelled accuracy = {:.4f}".format(idx,
                                                                                          class_loss_tracker[idx]/counter,
                                                                                          class_acc_tracker[idx]/counter))
            
            logger.info("div loss = {}".format(div_loss_tracker/counter))
            logger.info("ensemble loss = {}".format(total_loss_tracker/counter))
        
        return total_loss_tracker/counter
        
    def predict(self, x):
        self.classifier.eval()
        
        if self.use_gpu:
            x = x.to("cuda")
        
        with torch.no_grad():
            pred_y = self.classifier(x)
        
        return pred_y
        
    def predict_idx(self, x, idx):
        self.classifier.eval()
        
        if self.use_gpu:
            x = x.to("cuda")
        
        with torch.no_grad():
            pred_y = self.classifier(x)
        
        
        return pred_y[:,idx,:]
    