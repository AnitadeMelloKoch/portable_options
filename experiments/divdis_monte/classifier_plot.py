import os 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
import pandas as pd

divdis = [
    [0.7215, 0.9680, 0.9818, 0.9837],
    [0.66, 0.9461, 0.9831, 0.9846],
    [0.7203, 0.9734, 0.9804, 0.9836],
    [0.7432, 0.9747, 0.9826, 0.9852],
    [0.7591, 0.9729, 0.9842, 0.9836],
    [0.7270, 0.9533, 0.9811, 0.9838],
    [0.6519, 0.9774, 0.9832, 0.9811],
    [0.7117, 0.9672, 0.9738, 0.9826],
    [0.6322, 0.9317, 0.9824, 0.9836],
    [0.7211, 0.9626, 0.9852, 0.9835],
]

no_div = [
    [0.5985, 0.9716, 0.9779, 0.9831, ],
    [0.6228, 0.9664, 0.9846, 0.9847, ],
    [0.6586, 0.9726, 0.9820, 0.9843, ],
    [0.7591, 0.9526, 0.9797, 0.9846, ],
    [0.7451, 0.9654, 0.9807, 0.9867, ],
    [0.7303, 0.9697, 0.9848, 0.9856, ],
    [0.7298, 0.9534, 0.9802, 0.9832, ],
    [0.6395, 0.9106, 0.9824, 0.9843, ],
    [0.7667, 0.9418, 0.9796, 0.9842, ],
    [0.7366, 0.9576, 0.9785, 0.9844, ],
]

one_head = [
    [0.4429, 0.9597, 0.9824, 0.9657],
    [0.5872, 0.9065, 0.9661, 0.9811],
    [0.5931, 0.8671, 0.9560, 0.9815],
    [0.4575, 0.9597, 0.9847, 0.9671],
    [0.4302, 0.9578, 0.9742, 0.9644],
    [0.5827, 0.8354, 0.9591, 0.9784],
    [0.5858, 0.9484, 0.9833, 0.9815],
    [0.5439, 0.9652, 0.9755, 0.9807],
    [0.5187, 0.9640, 0.9685, 0.9785],
    [0.4047, 0.9561, 0.9723, 0.9829],
]

div_avg, div_std = np.mean(divdis, axis = 0), np.std(divdis, axis = 0)
nodiv_avg, nodiv_std = np.mean(no_div, axis=0), np.std(no_div, axis=0)
one_avg, one_std = np.mean(one_head, axis=0), np.std(one_head, axis=0)

data_dict = []


for seed_idx, seed in enumerate(one_head):
    for data_idx, data in enumerate(seed):
        data_dict.append({
            "seed": seed_idx,
            "Accuracy": data,
            "Number of Seen Ladders": data_idx+1,
            "method": "CNN"
        })

for seed_idx, seed in enumerate(no_div):
    for data_idx, data in enumerate(seed):
        data_dict.append({
            "seed": seed_idx,
            "Accuracy": data,
            "Number of Seen Ladders": data_idx+1,
            "method": "Standard Ensemble"
        })

for seed_idx, seed in enumerate(divdis):
    for data_idx, data in enumerate(seed):
        data_dict.append({
            "seed": seed_idx,
            "Accuracy": data,
            "Number of Seen Ladders": data_idx+1,
            "method": "D-BAT Ensemble"
        })



df = pd.DataFrame.from_dict(data_dict)

df = df[df["Number of Seen Ladders"] < 5]

sns.set(font_scale=1.1)
sns.set_style("white")

plot = sns.lineplot(data=df, x="Number of Seen Ladders", y="Accuracy", 
                    hue="method", style="method", legend="full", markers=True)

handles, labels = plot.figure.axes[0].get_legend_handles_labels()
plot.figure.axes[0].legend(handles=handles[0:], labels=labels[0:])
plot.figure.axes[0].set_xticks([1,2,3,4])


plot.figure.savefig('classifier_acc.png')

