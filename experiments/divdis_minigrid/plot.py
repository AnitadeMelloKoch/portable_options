import os 
import glob 
import matplotlib.pyplot as plt 
import pickle 
import pandas as pd 
import numpy as np

import seaborn as sns

def get_df_from_csv(exp, legend_key):
    files = glob.glob('runs/minigrid_runs/*{}*'.format(exp))
    
    dfs = []
    for idx, file in enumerate(files):
        df = pd.read_csv(file, header=0)
        new_index = pd.RangeIndex(start=df["Step"].min(), stop=df["Step"].max() + 1)
        df["rolling_success"] = df["Value"].rolling(5000, min_periods=1).mean()
        df.drop(columns=["Wall time", "Value"])
        df = df.set_index('Step')['rolling_success'].reindex(new_index).interpolate().reset_index()
        df.columns=['Step', 'rolling_success']
        df["seed"] = idx
        df["exp"] = legend_key
        df = df[df["Step"] < 1500000]
        df = df[df["Step"]%100 == 0]
        dfs.append(df)
        

    comb_df = pd.concat(dfs)
    return comb_df

# data = None
# with open('runs/fast_meta_image_full_10_timeout/11/checkpoints/experiment_results.pkl', 'rb') as f:
#     data = pickle.load(f)

# df = pd.DataFrame.from_dict(data)
# df["action"] = df["action"].apply(lambda x: x.item())
# print(df.iloc[:50]["action"])

# plot = sns.displot(df.iloc[:100]["action"].to_list(), bins=15)
# plot.figure.savefig("action_hist.png")

dqn_df = get_df_from_csv('dqn', "DQN")
ppo_df = get_df_from_csv('ppo', "PPO")
div_df = get_df_from_csv('div', "D-BAT Option Agent")

mega_df = pd.concat([dqn_df, ppo_df, div_df])

print(mega_df)

plot = sns.lineplot(data=mega_df, x="Step", y="rolling_success", hue="exp", err_style='band', errorbar='se')

handles, labels = plot.figure.axes[0].get_legend_handles_labels()
plot.figure.axes[0].legend(handles=handles[0:], labels=labels[0:])

plot.figure.axes[0].set_ylabel("Average Undiscounted Episode Reward")
plot.figure.axes[0].set_xlabel("Environment Steps")


plot.figure.savefig("minigrid_train.png")
