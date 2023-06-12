import os
import random
import numpy as np
import torch

from portable.option.sets.models.portable_set_decaying_div import EnsembleClassifierDecayingDiv
from evaluation.model_wrappers import EnsembleClassifierWrapper
from ..evaluators.utils import concatenate
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

class AttentionEvaluatorClassifier():

    def __init__(
            self,
            classifier,
            plot_dir,
            stack_size):
        
        self.classifier = classifier

        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.num_modules = self.classifier.num_modules

        self.true_data = torch.from_numpy(np.array([])).float()
        self.false_data = torch.from_numpy(np.array([])).float()
        self.stack_size = stack_size
        
        self.integrated_gradients = []
        for x in range(self.num_modules):
            self.integrated_gradients.append(NoiseTunnel(IntegratedGradients(
                EnsembleClassifierWrapper(
                    self.classifier,
                    x
                ))
            ))

    def add_true_from_files(self, file_list):
        for file in file_list:
            data = np.load(file)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
            self.true_data = concatenate(self.true_data, data)
        
    def add_false_from_files(self, file_list):
        for file in file_list:
            data = np.load(file)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
            self.false_data = concatenate(self.false_data, data)

    def evaluate(self, num_images):
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
    
    def _evaluate_images(self, images):
        images_list = []
        for image in images:
            vote_list = []
            single_image = image.unsqueeze(0)
            for idx in range(self.num_modules):
                pred = self.classifier.get_single_module(single_image, idx).detach().cpu().numpy()
                # print(pred)
                vote_list.append(pred)
            images_list.append(vote_list)
        return images_list
            

    def _plot_attributions(self, image, attributions, votes, plot_name):
        fig, axes = plt.subplots(nrows=self.num_modules+1, ncols=self.stack_size, figsize=(2*self.stack_size,2*(self.num_modules + 1)))
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
