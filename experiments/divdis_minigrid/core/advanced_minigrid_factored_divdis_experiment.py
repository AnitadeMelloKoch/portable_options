import logging 
import datetime 
import os 
import gin 
import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
from portable.utils.utils import set_seed 
from torch.utils.tensorboard import SummaryWriter
import torch

from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.memory import SetDataset

@gin.configurable 
class AdvancedMinigridFactoredDivDisExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 use_gpu,
                 
                 classifier_head_num,
                 classifier_learning_rate,
                 classifier_input_dim,
                 classifier_num_classes,
                 classifier_diversity_weight):
        
        self.seed = seed 
        self.base_dir = base_dir
        self.experiment_name = experiment_name 
        
        set_seed(seed)
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        self.classifier = DivDisClassifier(use_gpu=use_gpu,
                                           log_dir=self.log_dir,
                                           head_num=classifier_head_num,
                                           learning_rate=classifier_learning_rate,
                                           input_dim=classifier_input_dim,
                                           num_classes=classifier_num_classes,
                                           diversity_weight=classifier_diversity_weight)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        log_file = os.path.join(self.log_dir,
                                "{}.log".format(datetime.datetime.now()))
        
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.experiment_name, self.seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Seed: {}".format(seed))
        logging.info("Head num: {}".format(classifier_head_num))
        logging.info("Learning rate: {}".format(classifier_learning_rate))
        logging.info("Diversity weight: {}".format(classifier_diversity_weight))
    
    def save(self):
        self.classifier.save(path=self.save_dir)
    
    def load(self):
        self.classifier.load(path=self.save_dir)
    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        
        self.classifier.add_data(positive_files=positive_files,
                                 negative_files=negative_files,
                                 unlabelled_files=unlabelled_files)
    
    def train_classifier(self, epochs):
        self.classifier.train(epochs=epochs)
    
    def test_classifier(self,
                        test_positive_files,
                        test_negative_files):
        dataset = SetDataset(max_size=1e6,
                             batchsize=64)
        
        dataset.add_true_files(test_positive_files)
        dataset.add_false_files(test_negative_files)
        
        counter = 0
        accuracy = np.zeros(self.classifier.head_num)
        
        for _ in range(dataset.num_batches):
            counter += 1
            x, y = dataset.get_batch()
            pred_y = self.classifier.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
                
        return accuracy/counter
        
