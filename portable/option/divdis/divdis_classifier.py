import logging
from pyexpat import model 
import torch
import torch.nn as nn
import gin
import os
import numpy as np 
from tqdm import tqdm 

from portable.option.memory import SetDataset, UnbalancedSetDataset
from portable.option.divdis.models.mlp import MultiHeadMLP, OneHeadMLP
from portable.option.divdis.models.minigrid_cnn_16x16 import MinigridCNN16x16
# from portable.option.divdis.models.minigrid_cnn_large import MinigridCNNLarge
from portable.option.divdis.models.monte_cnn import MonteCNN
from portable.option.divdis.divdis import DivDisLoss
from portable.option.divdis.models.yolo import YOLOEnsemble
from portable.option.divdis.models.clip import Clip
logger = logging.getLogger(__name__)
MODEL_TYPE = [
    "one_head_mlp",
    "multi_head_mlp",
    "minigrid_cnn",
    "minigrid_large_cnn",
    "monte_cnn",
    "yolo",
    "clip"
]

@gin.configurable
class DivDisClassifier():
    def __init__(self,
                 use_gpu,
                 log_dir,
                 
                 head_num,
                 learning_rate,
                 num_classes,
                 diversity_weight,
                 
                 l2_reg_weight=0.001,
                 class_weight=None,
                 dataset_max_size=int(1e6),
                 dataset_batchsize=32,
                 unlabelled_dataset_batchsize=None,
                 
                 summary_writer=None,
                 model_name='yolo') -> None:
        

        if use_gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(use_gpu)) 
        
        self.dataset = SetDataset(max_size=dataset_max_size,
                                            batchsize=dataset_batchsize,
                                            unlabelled_batchsize=unlabelled_dataset_batchsize)
        self.learning_rate = learning_rate
        self.l2_reg_weight = l2_reg_weight
        
        self.head_num = head_num
        self.num_classes = num_classes
        
        self.log_dir = log_dir

        
        self.model_name = model_name
        self.reset_classifier()
        

        self.optimizer = torch.optim.Adam(self.classifier.parameters(),
                                          lr=learning_rate,
                                          weight_decay=l2_reg_weight # weight decay also works as L2 regularization
                                          )
        
        self.divdis_criterion = DivDisLoss(heads=head_num)
        
        if class_weight is not None:
            assert len(class_weight) == num_classes
            class_weight_tensor = torch.tensor(class_weight, dtype=torch.float).to(self.device)
        else:
            class_weight_tensor = torch.ones(num_classes, dtype=torch.float).to(self.device)
        
        self.ce_criterion = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
        self.diversity_weight = diversity_weight
        
        self.state_dim = 3
    
    def save(self, path):
        torch.save(self.classifier.state_dict(), os.path.join(path, 'classifier_ensemble.ckpt'))
        self.dataset.save(path)
    
    def load(self, path):
        if os.path.exists(os.path.join(path, 'classifier_ensemble.ckpt')):
            print("classifier loaded from: {}".format(path))
            self.classifier.load_state_dict(torch.load(os.path.join(path, 'classifier_ensemble.ckpt')))
            self.dataset.load(path)
    
    def set_class_weights(self, weights=None):
        if weights is None:
            weights = [0.5, 0.5] #self.dataset.get_equal_class_weight()
        
        self.ce_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights).to(self.device)
        )
    
    def reset_classifier(self):
        if self.model_name == "monte_cnn":
            self.classifier = MonteCNN(num_classes=self.num_classes,
                                       num_heads=self.head_num)
        elif self.model_name == "yolo":
            self.classifier = YOLOEnsemble(num_classes=self.num_classes,
                                               num_heads= self.head_num)
        elif self.model_name == "clip":
            self.classifier = Clip(num_classes=self.num_classes,
                                               num_heads= self.head_num)
        else:
            raise ValueError("model_name must be one of {}".format(MODEL_TYPE))
        self.classifier.to(self.device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.l2_reg_weight # weight decay also works as L2 regularization
                                          )
        
    
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

        # self.set_class_weights(self.dataset.get_equal_class_weight())
    
    
    def train(self,
              epochs,
              start_offset=0,
              progress_bar=False):

        # self.move_to_gpu()
        self.classifier.train()
        
        for epoch in tqdm(range(start_offset, start_offset+epochs), desc="Training Epochs", leave=False, disable=(not progress_bar)):
            self.dataset.shuffle()
            counter = 0
            
            class_loss_tracker = np.zeros(self.head_num)
            class_acc_tracker = np.zeros(self.head_num)
            div_loss_tracker = 0
            total_loss_tracker = 0
            
            self.dataset.shuffle()
            if self.dataset.unlabelled_data_length == 0:
                use_unlabelled_data = False
            else:
                use_unlabelled_data = True
            
            for _ in range(self.dataset.num_batches):
                counter += 1
                x, y = self.dataset.get_batch()
                if torch.sum(y) == 0 or torch.sum(y) == len(y):
                    continue
                if use_unlabelled_data:
                    unlabelled_x = self.dataset.get_unlabelled_batch()
                
                x = x.to(self.device)
                y = y.to(self.device)
                if use_unlabelled_data:
                    unlabelled_x = unlabelled_x.to(self.device)
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

                if use_unlabelled_data:
                    div_loss = self.divdis_criterion(unlabelled_pred)
                else:
                    div_loss = torch.tensor(0)
                
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
        
        if len(x.shape) == self.state_dim:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            pred_y = self.classifier(x)
        
        votes = torch.argmax(pred_y, axis=-1)
        
        votes = votes.cpu().numpy()
        self.votes = votes
        
        return pred_y, votes
        
    def predict_idx(self, x, idx):
        self.classifier.eval()
        
        x = x.to(self.device)
        
        with torch.no_grad():
            pred_y = self.classifier(x)
        

        return pred_y[:,idx,:]







