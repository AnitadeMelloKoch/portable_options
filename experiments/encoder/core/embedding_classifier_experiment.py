import logging 
import datetime 
import os 
import random 
import gin 
import torch 
import lzma 
import dill 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from portable.utils.utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from portable.option.ensemble.custom_attention import AutoEncoder
from experiments.encoder.core.classifier import Classifier

@gin.configurable
class EmbeddingClassifierExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 model_type,
                 input_size,
                 lr,
                 use_gpu):
        
        self.base_dir = base_dir
        self.name = experiment_name
        self.seed = seed
        set_seed(seed)
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.use_gpu = use_gpu
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        log_file = os.path.join(self.log_dir, 
                                "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))

        if self.use_gpu:
            self.embedding = AutoEncoder().to("cuda")
        else:
            self.embedding = AutoEncoder()
        self.embedding_loaded = False
        
        self.classifier = Classifier(model_type=model_type,
                                     input_size=input_size,
                                     lr=lr,
                                     use_gpu=use_gpu,
                                     embedding=self.embedding)
        
    def load_embedding(self, load_dir=None):
        if load_dir is None:
            load_dir = os.path.join(self.save_dir, 'embedding', 'model.ckpt')
        logging.info("[experiment embedding] Embedding loaded from {}".format(load_dir))
        self.embedding.load_state_dict(torch.load(load_dir))
        self.embedding_loaded = True

    def add_datafiles(self,
                      positive_files,
                      negative_files):
        
        self.classifier.add_data(positive_files=positive_files,
                                 negative_files=negative_files)
    
    def run(self,
            epochs):
        
        loss, accuracy, loss_val, accuracy_val = self.classifier.train(epochs=epochs)
        
        plt.plot(loss, label='training')
        plt.plot(loss_val, label='validation')
        
        plt.title("loss")
        plt.legend()
        
        plt.savefig(os.path.join(self.plot_dir, 'loss.png'))
        plt.cla()
        
        plt.plot(accuracy, label='training')
        plt.plot(accuracy_val, label='validation')
        
        plt.title("accuracy")
        plt.legend()
        
        plt.savefig(os.path.join(self.plot_dir, 'accuracy.png'))
        plt.cla()
        
        
        
    
    
    
    







