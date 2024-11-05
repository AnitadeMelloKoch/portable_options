import os 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
import pandas as pd

divdis = [
    [0.7215, 0.9680, 0.9818, 0.9837, 0.9869, 0.9893, 0.9914],
    [0.66, 0.9461, 0.9831, 0.9846, 0.9867, 0.9916, 0.9915],
    [0.7203, 0.9734, 0.9804, 0.9836, 0.9866, 0.9915, 0.9926]
]

no_div = [
    [0.5985, 0.9716, 0.9779, 0.9831, 0.9841, 0.9898, 0.9915],
    [0.6228, 0.9664, 0.9846, 0.9847, 0.9912, 0.9911, 0.9928],
    [0.6586, 0.9726, 0.9820, 0.9843, 0.9880, 0.9905, 0.9926]
]

one_head = [
    [0.4429, 0.9597, 0.9824, 0.9657, 0.9862, 0.9845, 0.9879],
    [0.5872, 0.9065, 0.9661, 0.9811, 0.9825, 0.9847, 0.9902],
    [0.5931, 0.8671, 0.9560, 0.9815, 0.9729, 0.9889, 0.9914]
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
            "method": "D-BAT Ensemble - no diversity"
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

plot = sns.lineplot(data=df, x="Number of Seen Ladders", y="Accuracy", hue="method", legend="full")

handles, labels = plot.figure.axes[0].get_legend_handles_labels()
plot.figure.axes[0].legend(handles=handles[0:], labels=labels[0:])
plot.figure.axes[0].set_xticks([1,2,3,4])


plot.figure.savefig('classifier_acc.png')

