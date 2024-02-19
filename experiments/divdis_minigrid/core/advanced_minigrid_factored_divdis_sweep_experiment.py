import logging 
import datetime 
import os 
import gin 
import numpy as np 
from portable.utils.utils import set_seed 
from torch.utils.tensorboard import SummaryWriter 
import torch 
from collections import deque
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from portable.option.memory import SetDataset
from portable.option.divdis.divdis_classifier import DivDisClassifier, transform

@gin.configurable
class FactoredAdvancedMinigridDivDisSweepExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 use_gpu,
                 default_epochs,
                 default_div_weight,
                 default_div_lr,
                 default_num_heads,
                 train_positive_files,
                 train_negative_files,
                 unlabelled_files,
                 test_positive_files,
                 test_negative_files):
        
        self.experiment_name = experiment_name
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name)
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.default_epochs = default_epochs
        self.default_div_weight = default_div_weight
        self.default_div_lr = default_div_lr
        self.default_num_heads = default_num_heads
        
        self.dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        self.dataset_negative = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        self.dataset_positive.set_transform_function(transform)
        self.dataset_negative.set_transform_function(transform)
        
        self.dataset_positive.add_true_files(test_positive_files)
        self.dataset_negative.add_false_files(test_negative_files)
        
        self.train_positive_files = train_positive_files
        self.train_negative_files = train_negative_files
        self.unlabelled_files = unlabelled_files
        
        log_file = os.path.join(self.log_dir, 
                                "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        
    
    def test_terminations(self,
                          dataset_positive,
                          dataset_negative,
                          classifier):
        
        counter = 0
        accuracy = np.zeros(classifier.head_num)
        accuracy_pos = np.zeros(classifier.head_num)
        accuracy_neg = np.zeros(classifier.head_num)
        
        for _ in range(dataset_positive.num_batches):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y = classifier.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_pos /= counter
        
        total_count = counter
        counter = 0
        
        for _ in range(dataset_negative.num_batches):
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y = classifier.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_neg /= counter
        total_count += counter
        
        accuracy /= total_count
        
        weighted_acc = (accuracy_pos + accuracy_neg)/2
        
        logging.info("============= Classifiers evaluated =============")
        for idx in range(classifier.head_num):
            logging.info("idx:{:.4f} true accuracy: {:.4f} false accuracy: {:.4f} total accuracy: {:.4f} weighted accuracy: {:.4f}".format(
                idx,
                accuracy_pos[idx],
                accuracy_neg[idx],
                accuracy[idx],
                weighted_acc[idx])
            )
        logging.info("=================================================")
        
        # accuracy: accuracy over entire dataset 
        # weighted_acc: accuracy of pos&neg weighted equally
        return accuracy_pos, accuracy_neg, accuracy, weighted_acc
    
    def save_results_dict(self,
                          save_dict,
                          file_name):
        
        with open(os.path.join(self.log_dir, file_name), 'wb') as f:
            pickle.dump(save_dict, f)
        
    
    def plot(self,
             plot_file,
             x_values,
             accuracies,
             losses,
             
             plot_title,
             x_label):
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        
        accuracies = np.array(accuracies)
        losses = np.array(losses)
        
        acc_mean = np.mean(accuracies, axis=1)
        acc_std = np.std(accuracies, axis=1)
        
        loss_mean = np.mean(losses, axis=1)
        loss_std = np.std(losses, axis=1)
        
        ax1.plot(x_values, loss_mean)
        ax1.fill_between(x_values, 
                         loss_mean-loss_std, 
                         loss_mean+loss_std,
                         alpha=0.2)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Final Training Loss')
        ax1.title.set_text('Loss')
        
        ax2.plot(x_values, acc_mean)
        ax2.fill_between(x_values,
                         acc_mean-acc_std,
                         acc_mean+acc_std,
                         alpha=0.2)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Test Accuracy')
        ax2.title.set_text('Accuracy')
        
        fig.suptitle(plot_title)
        fig.tight_layout()
        
        fig.savefig(plot_file)
        plt.close(fig)
    
    def sweep_class_div_weight(self,
                               start_weight,
                               end_weight,
                               num_samples,
                               num_seeds):
        
        # 1D array
        results_weight = []
        # 2D array for multiple seeds
        results_acc = []
        # 2D array for multiple seeds
        results_loss = []
        
        weights = np.linspace(start_weight, end_weight, num_samples)
        
        for weight in tqdm(weights, desc="weights", position=0):
            results_weight.append(weight)
            weight_acc = []
            weight_loss = []
            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr)
                
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                weight_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                
                weight_acc.append(max(acc))
            
            results_acc.append(weight_acc)
            results_loss.append(weight_loss)

        self.plot(os.path.join(self.plot_dir, "div_weight_sweep.png"),
                  results_weight,
                  results_acc,
                  results_loss,
                  "Sweep over Diversity Weight",
                  "Diversity Weight")
        
        save_dict = {"weights": results_weight,
                     "accuracies": results_acc,
                     "losses": results_loss}
        
        self.save_results_dict(save_dict,
                               "weights_sweep.pkl")
    
    def sweep_epochs(self,
                     start_epochs,
                     end_epochs,
                     step_size,
                     num_seeds):
        
        # 1D array
        results_epoch = []
        # 2D array for multiple seeds
        results_acc = []
        # 2D array for multiple seeds
        results_loss = []
        
        
        for epochs in tqdm(range(start_epochs, end_epochs+1, step_size), desc="epochs", position=0):
            results_epoch.append(epochs)
            epoch_acc = []
            epoch_loss = []
            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr)
                
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(epochs)
                epoch_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                
                epoch_acc.append(max(acc))
            
            results_acc.append(epoch_acc)
            results_loss.append(epoch_loss)
            
        self.plot(os.path.join(self.plot_dir, "epoch_sweep.png"),
                  results_epoch,
                  results_acc,
                  results_loss,
                  "Sweep over Train Epochs",
                  "Train Epochs")
        
        save_dict = {"epochs": results_epoch,
                     "accuracies": results_acc,
                     "losses": results_loss}
        
        self.save_results_dict(save_dict,
                               "epochs_sweep.pkl")
    
    def sweep_ensemble_size(self,
                            start_size,
                            end_size,
                            num_seeds):
        # 1D array
        results_size = []
        # 2D array for multiple seeds
        results_acc = []
        # 2D array for multiple seeds
        results_loss = []
        
        for num_heads in tqdm(range(start_size, end_size+1), desc="size", position=0):
            results_size.append(num_heads)
            size_acc = []
            size_loss = []
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=num_heads,
                                              learning_rate=self.default_div_lr)
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                size_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                size_acc.append(max(acc))
            results_acc.append(size_acc)
            results_loss.append(size_loss)
        
        self.plot(os.path.join(self.plot_dir, "ensemble_size_sweep.png"),
                  results_size,
                  results_acc,
                  results_loss,
                  "Sweep over Ensemble Size",
                  "No. Ensemble Members")
        
        save_dict = {"size": results_size,
                     "accuracies": results_acc,
                     "losses": results_loss}
        self.save_results_dict(save_dict,
                               "size_sweep.pkl")
    
    def sweep_div_batch_size(self,
                             start_batchsize,
                             end_batchsize,
                             batch_stepsize,
                             num_seeds):
        # 1D array
        results_batchsize = []
        # 2D array for multiple seeds
        results_acc = []
        # 2D array for multiple seeds
        results_loss = []
        
        for batch_size in tqdm(range(start_batchsize, end_batchsize+1, batch_stepsize), desc="batch_sizes", position=0):
            results_batchsize.append(batch_size)
            batch_acc = []
            batch_loss = []
            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr,
                                              unlabelled_dataset_batchsize=batch_size)
                
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                batch_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                batch_acc.append(max(acc))
            results_acc.append(batch_acc)
            results_loss.append(batch_loss)
        self.plot(os.path.join(self.plot_dir, "batch_sweep.png"),
                  results_batchsize,
                  results_acc,
                  results_loss,
                  "Sweep over Unlabelled Dataset Batchsize",
                  "Batchsize")
        
        save_dict = {"batchsizes": results_batchsize,
                     "accuracies": results_acc,
                     "losses": results_loss}
        
        self.save_results_dict(save_dict,
                               "batch_sweep.pkl")
    
    def sweep_lr(self,
                 start_lr,
                 end_lr,
                 num_samples,
                 num_seeds):
        # 1D array
        results_lr = []
        # 2D array for multiple seeds
        results_acc = []
        # 2D array for multiple seeds
        results_loss = []
        
        lrs = np.linspace(start_lr, end_lr, num_samples)
        
        for lr in tqdm(lrs, desc="lrs", position=0):
            results_lr.append(lr)
            lr_acc = []
            lr_loss = []
            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=lr)
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                lr_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                lr_acc.append(max(acc))
            results_acc.append(lr_acc)
            results_loss.append(lr_loss)
        
        self.plot(os.path.join(self.plot_dir, "lr_sweep.png"),
                  results_lr,
                  results_acc,
                  results_loss,
                  "Sweep over Learning Rate",
                  "Learning Rate")
    
    
    
    
    
    
    
    
    
    
    
    
    