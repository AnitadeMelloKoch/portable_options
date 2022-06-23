from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt 


def count(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


@count
def plot_attention_diversity(embedding, num_attentions=8, save_dir=None, plot_freq=1):
    """
    visualize whether embedding of each attention is getting more and more diverse
    """
    assert len(embedding) == num_attentions
    assert embedding[0][0, :, :, :].shape == (64, 10, 10), embedding[0].shape
    for i in range(num_attentions):
        plt.subplot(2, 4, i+1)
        plt.imshow(np.mean(embedding[i].cpu().detach().numpy(), axis=(0, 1)))
        plt.title("attention {}".format(i))
    # show/save fig
    if save_dir is not None:
        if plot_attention_diversity.calls % plot_freq == 0:
            path = Path(save_dir) / f"attention_diversity_{plot_attention_diversity.calls}.png"
            plt.savefig(path)
    else:
        plt.show()
    plt.close()
