import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from captum.attr import visualization as viz
import numpy as np

def plot_state_and_attributions(state, attribution, save_path):
    # assuming states are all 4 stacks
    fig = plt.figure(clear=True)    
    fig, axes = plt.subplots(nrows=3, ncols=4)
    
    state = state.numpy()
        
    for idx, ax in enumerate(axes[0,:]):
        ax.set_axis_off()
        ax.imshow(np.transpose(state[:,idx,:,:], (1,2,0)), cmap='gray')
    
    for idx, ax in enumerate(axes[1,:]):
        fig, ax = viz.visualize_image_attr(
            np.expand_dims(attribution[idx,:,:], axis=-1),
            np.transpose(state[:,idx,:,:], (1,2,0)),
            "masked_image",
            "positive",
            plt_fig_axis=(fig, ax)
        )
        
    for idx, ax in enumerate(axes[2,:]):
        fig, ax = viz.visualize_image_attr(
            np.expand_dims(attribution[idx,:,:], axis=-1),
            np.transpose(state[:,idx,:,:], (1,2,0)),
            "heat_map",
            "positive",
            plt_fig_axis=(fig, ax)
        )
        
    fig.savefig(save_path)

def concatenate(arr1, arr2):
    if len(arr1) == 0:
        return arr2
    else:
        return torch.cat((arr1, arr2), axis=0)
