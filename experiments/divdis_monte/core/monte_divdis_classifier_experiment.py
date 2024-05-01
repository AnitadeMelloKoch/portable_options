import datetime
import logging
import os
import pickle

import gin
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.memory import SetDataset
from portable.utils.utils import set_seed

def save_image(img, save_dir, batch_number, img_number_within_batch):
            # Construct filename to include both batch number and image number within the batch
            filename = f"{save_dir}/batch_{batch_number}_image_{img_number_within_batch}.png"
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for i, ax in enumerate(axes.flat):
                ax.imshow(img[i], cmap='gray')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)

def worker_initializer():
    plt.switch_backend('Agg')

@gin.configurable 
class MonteDivDisClassifierExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 use_gpu,
                 
                 classifier_input_dim,
                 classifier_num_classes,
                 
                 classifier_head_num,
                 classifier_learning_rate,
                 classifier_diversity_weight,
                 classifier_l2_reg_weight,
                 classifier_train_epochs):
        
        self.seed = seed 
        self.base_dir = base_dir
        self.experiment_name = experiment_name 
        self.train_epochs = classifier_train_epochs
        
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
                                           diversity_weight=classifier_diversity_weight,
                                           l2_reg_weight=classifier_l2_reg_weight,
                                           model_name='monte_cnn'
                                           )
        
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
        logging.info("L2 reg weight: {}".format(classifier_l2_reg_weight))
        logging.info("Train epochs: {}".format(classifier_train_epochs))
    
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
    
    def train_classifier(self):
        self.classifier.train(epochs=self.train_epochs)
    
    def test_classifier(self,
                        test_positive_files,
                        test_negative_files,
                        save=False):
        dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        dataset_negative = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        # dataset_positive.set_transform_function(transform)
        # dataset_negative.set_transform_function(transform)
        
        dataset_positive.add_true_files(test_positive_files)
        dataset_negative.add_false_files(test_negative_files)
    
        counter = 0
        accuracy = np.zeros(self.classifier.head_num)
        accuracy_pos = np.zeros(self.classifier.head_num)
        accuracy_neg = np.zeros(self.classifier.head_num)
        
        #for _ in range(dataset_positive.num_batches):
        for _ in range(3):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y = self.classifier.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)

                # save false positive images to file'
                if save:
                    save_dir = os.path.join(self.plot_dir, 'false_negative', 'head_{}'.format(idx))
                    os.makedirs(save_dir, exist_ok=True)
                    
                    false_neg_idx = (pred_class != y)
                    false_neg_imgs = x[false_neg_idx]
                    with Pool(initializer=worker_initializer) as pool:
                        args = [(img, save_dir, counter, i + 1) for i, img in enumerate(false_neg_imgs)]
                        pool.starmap(save_image, args)


        accuracy_pos /= counter
        
        total_count = counter
        counter = 0
        
        #for _ in range(dataset_negative.num_batches):
        for _ in range(3):    
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y = self.classifier.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)

                if save:
                    save_dir = os.path.join(self.plot_dir, 'false_positive', 'head_{}'.format(idx))
                    os.makedirs(save_dir, exist_ok=True)
                    
                    false_pos_idx = (pred_class != y)
                    false_pos_imgs = x[false_pos_idx]
                    with Pool(initializer=worker_initializer) as pool:
                        args = [(img, save_dir, counter, i + 1) for i, img in enumerate(false_pos_imgs)]
                        pool.starmap(save_image, args)
                    
        
        accuracy_neg /= counter
        total_count += counter
        
        accuracy /= total_count
        
        weighted_acc = (accuracy_pos + accuracy_neg)/2
        
        logging.info("============= Classifiers evaluated =============")
        for idx in range(self.classifier.head_num):
            logging.info("Head idx:{:<4}, True accuracy: {:.4f}, False accuracy: {:.4f}, Total accuracy: {:.4f}, Weighted accuracy: {:.4f}".format(
                idx,
                accuracy_pos[idx],
                accuracy_neg[idx],
                accuracy[idx],
                weighted_acc[idx])
            )
        logging.info("=================================================")
        
        return accuracy, weighted_acc
    
    def explain_classifiers(self,
                            test_data,
                            test_head):
        dataset = SetDataset(max_size=1e6,
                             batchsize=64)
                
        dataset.add_true_files(test_data)
        
        true_data = []
        false_data = []
        
        for _ in range(dataset.num_batches):
            x, _ = dataset.get_batch()
            pred_y = self.classifier.predict(x)
            pred_y = pred_y.cpu()
            
            pred_class = torch.argmax(pred_y[:,test_head,:], dim=1).detach()
            
            true_data += x[pred_class == 1]
            false_data += x[pred_class == 0]
        
        if len(true_data) != 0:
            true_data = torch.stack(true_data)
        else:
            true_data = torch.zeros(1,1)
        if len(false_data) != 0:
            false_data = torch.stack(false_data)
        else:
            false_data = torch.zeros(1,1)
        
        return torch.std_mean(true_data, dim=0), torch.std_mean(false_data, dim=0)
        
        
    
    
