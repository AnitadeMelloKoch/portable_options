from cgi import test
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.memory.set_dataset import SetDataset

from portable.option.sets.models.portable_set_decaying_div import EnsembleClassifierDecayingDiv
from evaluation.model_wrappers import EnsembleClassifierWrapper
from ..evaluators.utils import concatenate

from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import visualization as viz



class DivDisEvaluatorClassifier():

    def __init__(
            self,
            classifier,
            plot_dir=None,
            image_input=False,
            stack_size=2):
        
        self.classifier = classifier

        self.plot_dir = plot_dir
        if self.plot_dir:
            os.makedirs(self.plot_dir, exist_ok=True)
        
        self.head_num = self.classifier.head_num
        self.image_input = image_input

        self.test_dataset = SetDataset(max_size=1e6, batchsize=int(1e6))
        
        self.stack_size = stack_size
        
        self.integrated_gradients = [NoiseTunnel(IntegratedGradients(self.classifier.classifier.model[i])) for i in range(self.head_num)]
        self.ig_attr_test = [dict() for _ in range(self.head_num)]
        self.confusion_matrices = [None for _ in range(self.head_num)]
        self.classification_reports = [None for _ in range(self.head_num)]


    def add_test_files(self, true_files, false_files):
        self.test_dataset.add_true_files(true_files)
        self.test_dataset.add_false_files(false_files)

        
    def evaluate(self, num_sample, num_features=26):
        if self.image_input is False:
            states, labels = self.test_dataset.get_batch() # should be all data, since batchsize is max_size
            if self.classifier.use_gpu:
                states = states.to('cuda')
                labels = labels.to('cuda')
            if num_sample < len(states):
                rand_idx = random.sample(range(len(states)), num_sample)
                states = states[rand_idx]
                labels = labels[rand_idx]            
            with torch.no_grad():
                labels_pred = self.classifier.predict(states) # (batch_size, head_num, num_classes)
                labels_pred = torch.argmax(labels_pred, dim=2) # (batch_size, head_num)

            for head_idx in range(self.head_num):
                # get the predictions for the head
                labels_pred_head = labels_pred[:, head_idx]
                
                # add sklearn confusion matrix and classification report
                cm = confusion_matrix(labels.to('cpu').numpy(), labels_pred_head.to('cpu').numpy())
                self.confusion_matrices[head_idx] = cm
                
                report = classification_report(labels.to('cpu').numpy(), 
                                               labels_pred_head.to('cpu').numpy(),
                                               output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                self.classification_reports[head_idx] = report_df

                # attributions
                states_tp = states[(labels == 1) & (labels_pred_head == 1)]
                states_tn = states[(labels == 0) & (labels_pred_head == 0)]
                states_fp = states[(labels == 0) & (labels_pred_head == 1)]
                states_fn = states[(labels == 1) & (labels_pred_head == 0)]

                self.ig_attr_test[head_idx]['all'] = self.integrated_gradients[head_idx].attribute(states, target=labels_pred_head)
                self.ig_attr_test[head_idx]['true positive'] = self.integrated_gradients[head_idx].attribute(states_tp, target=1) if len(states_tp) > 0 else None
                self.ig_attr_test[head_idx]['true negative'] = self.integrated_gradients[head_idx].attribute(states_tn, target=0) if len(states_tn) > 0 else None
                self.ig_attr_test[head_idx]['false positive'] = self.integrated_gradients[head_idx].attribute(states_fp, target=1) if len(states_fp) > 0 else None
                self.ig_attr_test[head_idx]['false negative'] = self.integrated_gradients[head_idx].attribute(states_fn, target=0) if len(states_fn) > 0 else None
            
            self._plot_attributions_factored(num_features)
            self._plot_cm_cr()

        if self.image_input is True:
            raise NotImplementedError

    '''
    def evaluate_old(self, num_images):
        idxs = list(range(len(self.true_data)))

        random_true_images_idxs = random.sample(idxs, num_images)
        true_images = self.true_data[random_true_images_idxs]

        # self.classifier.get_votes(true_images)

        positive_attributions = self._attributions(
            true_images,
            self.integrated_gradients,
            1
        )
        positive_votes = self._evaluate_images(true_images)
        
        for idx, image in enumerate(true_images):
            # print("true image {}".format(idx))
            # print(positive_attributions[idx])
            self._plot_attributions(image,
                                    positive_attributions[idx],
                                    positive_votes[idx],
                                    "true_{}_ig.png".format(idx))
        
        idxs = list(range(len(self.false_data)))
        
        random_false_images_idxs = random.sample(idxs, num_images)
        false_images = self.false_data[random_false_images_idxs]
        
        false_attributions = self._attributions(
            false_images,
            self.integrated_gradients,
            0
        )
        false_votes = self._evaluate_images(false_images)

        for idx, image in enumerate(false_images):
            # print("false image {}".format(idx))
            # print(false_attributions[idx])
            self._plot_attributions(image,
                                    false_attributions[idx],
                                    false_votes[idx],
                                    "false_{}_ig.png".format(idx))

    def _attributions(self, images, integrated_gradients, target):
        attributions = []
        for image in images:
            single_image = image.unsqueeze(0)
            image_attr = []
            for ig in integrated_gradients:
                image_attr.append(ig.attribute(
                    single_image,
                    nt_samples=10,
                    target=target,
                    n_steps=10
                ))
            attributions.append(image_attr)

        return attributions

    def _attributions_factored(self, states, integrated_gradients, target):
        attributions = []
        for state in states:
            single_state = state.unsqueeze(0)
            state_attr = []
            for ig in integrated_gradients:
                state_attr.append(ig.attribute(
                    single_state,
                    nt_samples=10,
                    target=target,
                    n_steps=10
                ))
            attributions.append(image_attr)

        return attributions
    
    def _evaluate_images(self, images):
        images_list = []
        for image in images:
            vote_list = []
            single_image = image.unsqueeze(0)
            for idx in range(self.head_num):
                pred = self.classifier.get_single_module(single_image, idx).detach().cpu().numpy()
                # print(pred)
                vote_list.append(pred)
            images_list.append(vote_list)
        return images_list      

    def _plot_attributions(self, image, attributions, votes, plot_name):
        fig, axes = plt.subplots(nrows=self.head_num+1, ncols=self.stack_size, figsize=(2*self.stack_size,2*(self.head_num + 1)))
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        image = image.numpy()
        
        for idx, ax in enumerate(axes[0,:]):
            ax.set_axis_off()
            ax.imshow(image[idx,:,:], cmap='gray')
        
        pad = 5
        for ax, vote in zip(axes[1:,0], votes):
            ax.annotate(np.argmax(vote), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        fontsize=10, ha='right', va='center', rotation=90)
            
        for idx, attribute in enumerate(attributions):
            print(attribute)
            attribute = attribute.cpu().detach().numpy()
            for ax_idx, ax in enumerate(axes[idx+1,:]):
                if not np.sum(attribute[:,ax_idx,...]) == 0:
                    fig, ax = viz.visualize_image_attr(
                        np.transpose(attribute[:,ax_idx,...], (1,2,0)),
                        np.expand_dims(image[ax_idx,:,:], axis=-1),
                        "heat_map",
                        "positive",
                        plt_fig_axis=(fig,ax)
                    )
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(
                        np.transpose(np.zeros_like(attribute[:,ax_idx,...]),
                                     (1,2,0)))
        plt.tight_layout()
        savepath = os.path.join(self.plot_dir, plot_name)
        fig.savefig(savepath)
        plt.close(fig)
    '''

    def _plot_attributions_factored(self, num_features=26):

        fig, axes = plt.subplots(nrows=self.head_num+0, ncols=5, figsize=(30,5*(self.head_num)), sharey='all')
        if self.head_num == 1:  # Ensure axes is iterable when there's only one subplot
            axes = [axes]
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)

        x_axis_data = np.arange(26)
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
                ax = axes[i, j]
                ig_attr_test = ig_attr_head[attr_type]
                
                if ig_attr_test is not None:  # Check if attribution was calculated
                    ig_attr_test_sum = ig_attr_test.detach().to('cpu').numpy().sum(0)
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
        self.test_dataset = SetDataset(max_size=1e6, batchsize=int(1e6))


    def get_head_complexity(self):
        head_complexity = []
        for i in range(self.head_num):
            ig_attr_test = self.ig_attr_test[i]['all']
            ig_attr_test_std = ig_attr_test.detach().to('cpu').numpy().std(0)
            head_complexity.append(ig_attr_test_std.mean())
        return head_complexity