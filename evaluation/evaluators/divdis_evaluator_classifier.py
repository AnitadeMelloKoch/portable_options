import datetime
import logging
import os
import pickle
import random
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.memory.set_dataset import SetDataset

from portable.option.sets.models.portable_set_decaying_div import EnsembleClassifierDecayingDiv
from evaluation.model_wrappers import EnsembleClassifierWrapper
from .utils import concatenate

from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import visualization as viz



class DivDisEvaluatorClassifier():

    def __init__(
            self,
            classifier,
            image_input=True,
            test_batch_size=64,
            base_dir=None,
            stack_size=4):
        
        self.classifier = classifier

        self.base_dir = base_dir
        self.plot_dir = os.path.join(self.base_dir,'plots')
        self.log_dir = os.path.join(self.base_dir,'logs')
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.head_num = self.classifier.head_num
        self.image_input = image_input

        self.positive_test_files = []
        self.negative_test_files = []

        self.test_batch_size = test_batch_size
        self.test_dataset = SetDataset(max_size=1e6, batchsize=self.test_batch_size)
        
        self.stack_size = stack_size
        
        # self.integrated_gradients = [NoiseTunnel(IntegratedGradients(self.classifier.classifier.full_model[i])) for i in range(self.head_num)]
        self.integrated_gradients = [(DeepLift(self.classifier.classifier.full_model[i])) for i in range(self.head_num)]
        self.ig_attr_test = [dict() for _ in range(self.head_num)]
        self.confusion_matrices = [None for _ in range(self.head_num)]
        self.classification_reports = [None for _ in range(self.head_num)]

        self.writer = SummaryWriter(log_dir=self.log_dir)
        log_file = os.path.join(self.log_dir,
                                "{}.log".format(datetime.datetime.now()))
        
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        



    def evaluate_images(self, num_images=5):
        images, labels = self.test_dataset.get_batch()
        
        images = images.to(self.classifier.device)
        labels = labels.to(self.classifier.device)


        #colors = ['#FF0000', '#FF4500', '#808080', '#90EE90', '#00FF00']
        #colors = ['#FF00FF', '#FF1493', '#303030', '#00FF00', '#39FF14']
        colors = ['#FF0044', '#FF3030', '#303030', '#00FF00', '#39FF14']
        n_bins = 100
        custom_cmap = LinearSegmentedColormap.from_list('custom_saliency', colors, N=n_bins)

            
        for image_idx in tqdm(range(num_images), desc='Evaluating Images'):
            image, label = images[image_idx].unsqueeze(0), labels[image_idx].item()

            image.requires_grad_()
            
            # Create a figure with subplots
            #fig, axes = plt.subplots(nrows=self.head_num+1, ncols=self.stack_size, figsize=(5*self.stack_size, 5*self.head_num))
            fig, axes = plt.subplots(nrows=self.head_num+0, ncols=self.stack_size, figsize=(3.7*self.stack_size, 4.3*(self.head_num+0)))
            fig.suptitle(f'True Label: {"Positive" if label else "Negative"}', 
                         ha='center', fontsize=20, fontweight='bold')

            predictions, votes = self.classifier.predict(image)  # Get predictions for the current head
            predicted_labels = predictions.squeeze().argmax(dim=-1)  # Assuming single label prediction

            #for i in range(self.stack_size):
            #    ax = axes[0, i]
            #    ax.imshow(image[0, i].detach().cpu().numpy(), cmap='gray')
            #    ax.set_xticks([])
            #    ax.set_yticks([])
            #axes[0, 0].text(-0.1, 0.5, "Original Image", transform=axes[0, 0].transAxes, 
            #    va='center', ha='center', fontsize=12, rotation='vertical')
                
            # Loop through each classifier head
            nonagreement = False
            for head_idx in range(self.head_num):
                pred_label_head = predicted_labels[head_idx].detach().cpu().numpy()
                # print("head idx:", head_idx)
                # print("pred label head:", pred_label_head)
                # print("dim:", pred_label_head.shape)
                # print("integrated grad:", self.integrated_gradients[head_idx])
                # print("image_shape:", image.shape)
                # print("label:", label)
                # print("attr dimension:", self.integrated_gradients[head_idx].attribute(
                #     image,
                #     target=label
                # ).squeeze().cpu().detach().numpy().shape)

                #attr = self.integrated_gradients[head_idx].attribute(
                #    image,
                #    nt_samples=10,
                #    n_steps=10,
                #    target=label
                #).squeeze().cpu().detach().numpy().transpose(1, 2, 0) # (H, W, C)
                attr = self.integrated_gradients[head_idx].attribute(
                    image,
                    target=label
                ).squeeze().cpu().detach().numpy().transpose(1, 2, 0) # (H, W, C)
                
                display_image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0) # (H, W, C)

                for channel_idx in range(self.stack_size):
                    ax = axes[head_idx+0, channel_idx]
                    ax.imshow(display_image[:,:,channel_idx], cmap='gray')
                    
                    # Visualize attributions with heatmap
                    fig, ax = viz.visualize_image_attr(
                        attr=np.expand_dims(attr[:,:,channel_idx], axis=-1), 
                        original_image=np.expand_dims(display_image[:,:,channel_idx], axis=-1), 
                        method='blended_heat_map', sign='all', alpha_overlay=0.7, cmap=custom_cmap,
                        show_colorbar=False, plt_fig_axis=(fig, ax),
                        use_pyplot=False
                    )
                    ax.set_axis_off()
                

                if(label == 1) & (pred_label_head == 1):
                    row_name = ('True Positive')
                elif(label == 0) & (pred_label_head == 0):
                    row_name = ('True Negative')
                elif(label == 0) & (pred_label_head == 1):
                    row_name = ('False Positive')
                elif(label == 1) & (pred_label_head == 0):
                    row_name = ('False Negative')

                if row_name == 'False Positive' or row_name == 'False Negative':
                    nonagreement = True
            
                #axes[head_idx,0].text(-0.2, 0.5, row_name, transform=axes[head_idx,0].transAxes, 
                #    va='center', ha='right', fontsize=12, fontweight='bold')
                axes[head_idx+0,0].text(-0.1, 0.5, row_name, transform=axes[head_idx+0,0].transAxes, 
                    va='center', ha='center', fontsize=16, rotation='vertical')

                    
            plt.tight_layout()
            #plt.show()

            for row in range(1, self.head_num + 0):
                pos_prev = axes[row-1, 0].get_position()
                pos_next = axes[row, 0].get_position()
                y = (pos_prev.y0 + pos_next.y1) / 2
                
                line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color='black', linewidth=2)
                fig.add_artist(line)

            pickle_filename = os.path.join(self.plot_dir, f"image_{image_idx}.pkl")
            with open(pickle_filename, 'wb') as f:
                pickle.dump(fig, f)
                    
            # Optionally save the figure
            img_name =  f'{"nonagreeing_" if nonagreement else ""}image_{image_idx}.png'
            fig.savefig(os.path.join(self.plot_dir, img_name))
            plt.close(fig)
    

    def evaluate_factored(self, sample_size=0.1, plot=False):
        num_batches_sampled = int(self.test_dataset.num_batches*sample_size)
        num_test_states = self.test_dataset.batchsize * num_batches_sampled
        #all_states = np.zeros((num_test_states, num_features))

        all_labels = np.zeros((num_test_states, ))
        all_labels_pred = np.zeros((num_test_states, self.head_num))
        self.ig_attr_test = [{'all':[],
                              'true positive':[],
                              'false positive':[],
                              'true negative':[],
                              'false negative':[]} for _ in range(self.head_num)]

        self.test_dataset.shuffle()
        for i in range(num_batches_sampled): # attributions take too long, sample a subset of the test data
            #print(f'Batch {i+1}/{self.test_dataset.num_batches}')
            
            states, labels = self.test_dataset.get_batch() # should be all data, since batchsize is max_size
            i0, i1 = i*self.test_dataset.batchsize, (i+1)*self.test_dataset.batchsize

            all_labels[i0:i1] = labels.numpy()

            if self.classifier.use_gpu:
                states = states.to('cuda')
                labels = labels.to('cuda')
            
            labels_pred = self.classifier.predict(states).detach() # (batch_size, head_num, num_classes)
            labels_pred = torch.argmax(labels_pred, dim=-1) # (batch_size, head_num)
                
            for head_idx in range(self.head_num):
                # get the predictions for the head
                labels_pred_head = labels_pred[:, head_idx]
            
                # attributions
                states_tp = states[(labels == 1) & (labels_pred_head == 1)]
                states_tn = states[(labels == 0) & (labels_pred_head == 0)]
                states_fp = states[(labels == 0) & (labels_pred_head == 1)]
                states_fn = states[(labels == 1) & (labels_pred_head == 0)]

                self.ig_attr_test[head_idx]['all'           ].append(self.integrated_gradients[head_idx].attribute(
                    states,    nt_samples=10, n_steps=10, target=labels_pred_head).detach().to('cpu').numpy()) if len(states) > 0 else np.array([])
                self.ig_attr_test[head_idx]['true positive' ].append(self.integrated_gradients[head_idx].attribute(
                    states_tp, nt_samples=10, n_steps=10, target=1).detach().to('cpu').numpy()) if len(states_tp) > 0 else np.array([])
                self.ig_attr_test[head_idx]['true negative' ].append(self.integrated_gradients[head_idx].attribute(
                    states_tn, nt_samples=10, n_steps=10, target=0).detach().to('cpu').numpy()) if len(states_tn) > 0 else np.array([])
                self.ig_attr_test[head_idx]['false positive'].append(self.integrated_gradients[head_idx].attribute(
                    states_fp, nt_samples=10, n_steps=10, target=1).detach().to('cpu').numpy()) if len(states_fp) > 0 else np.array([])
                self.ig_attr_test[head_idx]['false negative'].append(self.integrated_gradients[head_idx].attribute(
                    states_fn, nt_samples=10, n_steps=10, target=0).detach().to('cpu').numpy()) if len(states_fn) > 0 else np.array([])

        # concatenate all attributions from all batches, independently for each head and each attribution type
        self.ig_attr_test = [{k: np.concatenate(v) if len(v)>0 else np.array([]) for k, v in ig_attr_head.items()} for ig_attr_head in self.ig_attr_test]

        if plot:
            for head_idx in range(self.head_num):
                labels_pred_head = all_labels_pred[:, head_idx]
                cm = confusion_matrix(all_labels, labels_pred_head)
                self.confusion_matrices[head_idx] = cm
            
                report = classification_report(all_labels, 
                                            labels_pred_head,
                                            output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                self.classification_reports[head_idx] = report_df

            self._plot_cm_cr()

            if not self.image_input:
                self._plot_attributions_factored(26)
        
    

    def _plot_attributions_factored(self, num_features=26):
        assert self.image_input is False, "Not implemented for image input"

        fig, axes = plt.subplots(nrows=self.head_num+0, ncols=5, figsize=(30,5*(self.head_num)), sharey='all')
        if self.head_num == 1:  # Ensure axes is iterable when there's only one subplot
            axes = [axes]
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)

        x_axis_data = np.arange(num_features)
        #x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))
        x_axis_data_labels = ['door_x', 'door_y', 'door_locked', 'door_open', 'door_color', 
                              'key1_x', 'key1_y', 'key1_color', 
                              'key2_x', 'key2_y', 'key2_color',
                              'key3_x', 'key3_y', 'key3_color',
                              'key4_x', 'key4_y', 'key4_color',
                              'key5_x', 'key5_y', 'key5_color',
                              'agent_x', 'agent_y', 'agent_dir', 
                              'goal_x', 'goal_y', 'split']

        for i in range(self.head_num):
            ig_attr_head = self.ig_attr_test[i]
            for j, attr_type in enumerate(['all','true positive','false positive','true negative','false negative']):
                ax = axes[i,j]
                ig_attr_test = ig_attr_head[attr_type]
                
                if len(ig_attr_test) > 0:  # Check if attribution was calculated, should be non-empty
                    ig_attr_test_sum = ig_attr_test.sum(0)
                    # normalize the sum of attributions by dividing by L1 norm
                    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1) if np.linalg.norm(ig_attr_test_sum, ord=1) > 0 else ig_attr_test_sum
                    if attr_type == 'all':
                        ax.bar(x_axis_data, ig_attr_test_norm_sum, width=0.6, align='center', alpha=0.8, color='#BB2233')
                        ax.set_title(f'Head {i+1}, all')
                    else:
                        ax.bar(x_axis_data, ig_attr_test_norm_sum, width=0.6, align='center', alpha=0.8, color='#3388EE')
                        ax.set_title(f'Head {i+1}, {attr_type.replace("_", " ").capitalize()}')
                    # ticks and labels
                    ax.set_xticks(x_axis_data)
                    ax.set_xticklabels(x_axis_data_labels, rotation='vertical')
                    ax.set_ylabel('Attributions')

                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.set_title(f'Head {i+1}, {attr_type.replace("_", " ").capitalize()}')
                    
                # grids behind bars
                ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
                ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
                ax.set_axisbelow(True)

        plt.tight_layout()
        plt.show()

        savepath = os.path.join(self.plot_dir, 'factored_attributions.png')
        fig.savefig(savepath)
        plt.close(fig)


    def _plot_cm_cr(self):
        fig, axes = plt.subplots(nrows=self.head_num, ncols=2, figsize=(8,3*(self.head_num)))
        if self.head_num == 1:  # Ensure axes is iterable when there's only one subplot
            axes = [axes]
        for i in range(self.head_num):
            cm = self.confusion_matrices[i]
            report_df = self.classification_reports[i]

            # Plotting Confusion Matrix
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=axes[i][0])
            axes[i][0].set_title(f'Head {i+1}, Confusion Matrix')
            axes[i][0].set_xlabel('Predicted labels')
            axes[i][0].set_ylabel('True labels')

            # Plotting Classification Report
            report_text = "\n".join([
                f'{index}: Precision={row["precision"]:.2f}, Recall={row["recall"]:.2f}, F1-Score={row["f1-score"]:.2f}, Support={row["support"]}'
                for index, row in report_df.iterrows()
                if index not in ['accuracy', 'macro avg', 'weighted avg']
            ])

            header_text = "Class Metrics:\n(Precision, Recall, F1-Score, Support)\n"
            report_text = header_text + report_text
            axes[i][1].axis('off')
            axes[i][1].text(0.01, 1, f'Head {i+1}, Classification Report:\n\n' + report_text, verticalalignment='top')

        plt.tight_layout()
        plt.show()

        # Saving the entire figure instead of individual plots
        savepath = os.path.join(self.plot_dir, 'cm_and_cr.png')
        fig.savefig(savepath)
        plt.close(fig)

            
    def reset_test_dataset(self):
        self.test_dataset = SetDataset(max_size=1e6, batchsize=self.test_batch_size)


    def get_head_complexity(self):
        """ call only after evaluate_factored() to get the head complexity"""
        head_complexity = []
        for i in range(self.head_num):
            ig_attr_test = self.ig_attr_test[i]['all']
            n = ig_attr_test.shape[0]
            ig_attr_test = ig_attr_test.reshape(n, -1)
            ig_attr_test_std = ig_attr_test.std(0)
            head_complexity.append(ig_attr_test_std.mean())
        return head_complexity


    def head_complexity(self, sample_size=0.1):
        """ call this directly to get the head complexity"""
        self.evaluate_factored(sample_size)
        return self.get_head_complexity()


    def add_test_files(self, true_files, false_files):
        self.positive_test_files = true_files
        self.negative_test_files = false_files
        
        self.test_dataset.add_true_files(true_files)
        self.test_dataset.add_false_files(false_files)

    def test_classifier(self):
        dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        dataset_negative = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        #dataset_positive.set_transform_function(transform)
        #dataset_negative.set_transform_function(transform)
        
        dataset_positive.add_true_files(self.positive_test_files)
        dataset_negative.add_false_files(self.negative_test_files)
    
        counter = 0
        accuracy = np.zeros(self.classifier.head_num)
        accuracy_pos = np.zeros(self.classifier.head_num)
        accuracy_neg = np.zeros(self.classifier.head_num)
        
        for _ in range(dataset_positive.num_batches):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y, votes = self.classifier.predict(x)
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach().cpu()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)

        accuracy_pos /= counter
        
        total_count = counter
        counter = 0
        
        for _ in range(dataset_negative.num_batches):
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y, votes = self.classifier.predict(x)
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach().cpu()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)

        
        accuracy_neg /= counter
        total_count += counter
        
        accuracy /= total_count
        
        weighted_acc = (accuracy_pos + accuracy_neg)/2
        
        '''logging.info("============= Classifiers evaluated =============")
        for idx in range(self.classifier.head_num):
            logging.info("Head idx:{:<4}, True accuracy: {:.4f}, False accuracy: {:.4f}, Total accuracy: {:.4f}, Weighted accuracy: {:.4f}".format(
                idx,
                accuracy_pos[idx],
                accuracy_neg[idx],
                accuracy[idx],
                weighted_acc[idx])
            )
        logging.info("=================================================")
        '''
        return accuracy_pos, accuracy_neg, accuracy, weighted_acc