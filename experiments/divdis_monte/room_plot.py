import os
import glob 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

def extract_room_seed(file, folder_idx):
    folders = file.split("/")
    folder_name = folders[folder_idx]
    split_folder_name = folder_name.split("_")
    
    room = split_folder_name[6]
    seed = split_folder_name[9]
    
    return room, seed

def get_rooms_seeds(files, folder_idx):
    rooms, seeds = [], []
    for file in files:
        room, seed = extract_room_seed(file, folder_idx)
        rooms.append(int(room))
        seeds.append(int(seed))
    
    return rooms, seeds

def get_data_files(base_dir, exp_str):
    files = glob.glob(base_dir + '/*' + exp_str + '*/*/*/checkpoints/experiment_data.pkl')
    
    return files

def get_df_from_pickle(file_name, num_rooms, seed):
    with open(file_name, 'rb') as f:
        experiment_data = pickle.load(f)

    exp_df = pd.DataFrame.from_dict(experiment_data)
    exp_df["rolling_success"] = exp_df["true_success"].rolling(100,  min_periods=1).mean()
    exp_df.insert(0, "num_rooms", num_rooms, True)
    exp_df["num_rooms"] = exp_df["num_rooms"].astype(int)
    exp_df.insert(1, "seed", seed, True)
    exp_df["seed"] = exp_df["seed"].astype(int)
    exp_df = exp_df.rename(columns={"idx":"head_idx"})
    
    return exp_df

def get_combined_df(files, room_nums, seeds):
    dfs = []
    for file, room_num, seed in zip(files, room_nums, seeds):
        dfs.append(get_df_from_pickle(file, room_num, seed))
    
    return pd.concat(dfs)

def scatter_from_terms(term_list, plot_dir):
    room_x = defaultdict(list)
    room_y = defaultdict(list)
    
    for loc in term_list:
        room_x[loc[2]].append(loc[0])
        room_y[loc[2]].append(loc[1])
    
    for room in room_x.keys():
        file_name = os.path.join(plot_dir, "room_{}.png".format(room))
        os.makedirs(plot_dir, exist_ok=True)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_ylim([0,300])
        ax.set_xlim([0,160])
        ax.scatter(room_x[room], room_y[room])
        fig.savefig(file_name)
        plt.close(fig)

def get_success_data_from_df(df, 
                             seen_rooms, 
                             seeds, 
                             env_idxs,
                             head_idxs):
    plot_points_avg = np.zeros(len(seen_rooms))
    plot_points_var = np.zeros(len(seen_rooms))
    for num_rooms in seen_rooms:
        seed_results = []
        for seed in seeds:
            env_results = []
            for env_idx in env_idxs:
                head_results = []
                for head_idx in head_idxs:
                    mini_df = df.loc[
                        (df['num_rooms']==num_rooms)&
                        (df['seed']==seed)&
                        (df['env_idx']==env_idx)&
                        (df['head_idx']==head_idx)]
                    if len(mini_df) == 0:
                        head_results.append(0)
                    else:
                        head_results.append(mini_df.iloc[-1]["rolling_success"])
                env_results.append(max(head_results))
            seed_results.append(np.mean(env_results))
        plot_points_avg[num_rooms-1] = np.mean(seed_results)
        plot_points_var[num_rooms-1] = np.std(seed_results)
    
    return plot_points_avg, plot_points_var

def room_success_by_seen_plot(df):
    head_idxs = df['head_idx'].unique()
    env_idxs = df['env_idx'].unique()
    rooms = df['num_rooms'].unique()
    seeds = df['seed'].unique()
    avg, std = get_success_data_from_df(df,
                                        rooms,
                                        seeds,
                                        env_idxs,
                                        head_idxs)
    
    print("===================")
    print(avg)
    print(std)
    print("===================")
    
def room_scatter_plots(df, rooms, seeds):
    head_idxs = df['head_idx'].unique()

file_dir = "runs/"

files = get_data_files(file_dir, "ladders")
rooms, seeds = get_rooms_seeds(files, 1)
df = get_combined_df(files, rooms, seeds)
room_success_by_seen_plot(df)




