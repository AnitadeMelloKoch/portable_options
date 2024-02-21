import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from ..evaluators.utils import concatenate
from evaluation.model_wrappers import EnsemblePolicyWrapper
from portable.utils import set_player_ram
#from portable.utils.video_generator import VideoGenerator

class AttentionEvaluatorPolicy():
    
    def __init__(self,
                 policy,
                 plot_dir,
                 stack_size=4):
        self.device = torch.device("cuda")
        
        self.policy = policy
        
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.num_modules = self.policy.num_modules
        
        self.integrated_gradients = []
        for x in range(self.num_modules):
            self.integrated_gradients.append(NoiseTunnel(
                IntegratedGradients(
                    EnsemblePolicyWrapper(
                        self.policy,
                        x
                    )
                )
            ))
        
        self.video_generator = VideoGenerator(base_path=plot_dir)
        self.episode_num = 0
        self.stack_size = stack_size
        
    def evaluate(self, env, state):
        
        self.video_generator.clear_images()
        done = False
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        while not done:
            action, all_actions, q_vals, _ = self.policy.predict_actions(state, True)
            state, reward, done, info = env.step(action)
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            attributions = self._attributions(state, self.integrated_gradients, all_actions)
            self._plot_attributions(state,
                                    attributions,
                                    all_actions,
                                    q_vals)
        
        # self.video_generator.create_video_from_images("episode_{}".format(self.episode_num))
        self.episode_num += 1
            
    def _attributions(self, state, integrated_gradients, chosen_actions):
        attributions = []
        
        for idx, ig in enumerate(integrated_gradients):
            attributions.append(ig.attribute(
                state,
                nt_samples=10,
                target=int(chosen_actions[idx]),
                n_steps=10
            ))
        
        return attributions
            
    def _plot_attributions(self, 
                           state, 
                           attributions, 
                           actions, 
                           q_values):
        fig, axes = plt.subplots(nrows=self.num_modules+1, 
                                 ncols=self.stack_size, 
                                 figsize=(2*self.stack_size,2*(self.num_modules+1)))

        image = state.squeeze().cpu().numpy()
        
        for idx, ax in enumerate(axes[0,:]):
            ax.set_axis_off()
            ax.imshow(image[idx,:,:], cmap='gray')
        
        pad = 5
        for ax, action, q_val in zip(axes[1:,0], actions, q_values):
            ax.annotate("{} ({:.2f})".format(action, q_val), 
                        xy=(0,0.5),xytext=(-ax.yaxis.labelpad-pad,0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        fontsize=10, ha='right', va='center', rotation=90)
        
        for idx, attribute in enumerate(attributions):
            attribute = attribute.cpu().detach().numpy()
            for ax_idx, ax in enumerate(axes[idx+1,:]):
                if not np.sum(attribute[:,ax_idx,...], (1,2,0)) == 0:
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
                        np.transpose(np.zeros_like(attribute[:,ax_idx,...])),
                        (1,2,0)
                    )
        plt.tight_layout()
        self.video_generator.save_image(fig)
        plt.close(fig)