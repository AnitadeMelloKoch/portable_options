import os
import glob 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

TERMS = [(76, 192, 1), (20, 148, 1), (133, 148, 1),
         (77,235,4),(77,235,10),
         (77, 235, 6),
         (77, 235, 11),(77, 235, 19),
         (77,235,13),(76, 235, 21),
         (77, 235, 22),
         (77, 235, 9)]

def get_dist_point(current_room, dest_room):
    ROOM_TOP = 253
    ROOM_BOT = 135
    ROOM_LEFT = 0
    ROOM_RIGHT = 150
    
    if dest_room == 1:
        if current_room == 0:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 1)
        elif current_room == 2:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 1)
        elif current_room == 4:
            return (77, ROOM_TOP), (77, ROOM_BOT, 0)
        elif current_room == 6:
            return (77, ROOM_TOP), (77, ROOM_BOT, 2)
        elif current_room == 9:
            return (77, ROOM_TOP), (77, ROOM_BOT, 3)
        elif current_room == 3:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 4)
        elif current_room == 5:
            return (77, ROOM_BOT), (77, ROOM_TOP, 11)
        elif current_room == 11:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 10)
        elif current_room == 10:
            return (77, ROOM_TOP), (77, ROOM_BOT, 4)
        elif current_room == 19:
            return (77, ROOM_TOP), (77, ROOM_BOT, 11)
        elif current_room == 11:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 10)
        elif current_room == 7:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 6)
        elif current_room == 13:
            return (77, ROOM_TOP), (77, ROOM_BOT, 7)
        elif current_room == 12:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 13)
        elif current_room == 21:
            return (77, ROOM_TOP), (77, ROOM_BOT, 13)
        elif current_room == 14:
            return (77, ROOM_BOT), (77, ROOM_TOP, 22)
        elif current_room == 22:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 21)
        elif current_room == 18:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 19)
        elif current_room == 20:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 19)
        elif current_room == 23:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 22)
        elif current_room == 8:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 9)
    
    if dest_room == 6:
        if current_room == 1:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 2)
        elif current_room == 2:
            return (77, ROOM_BOT), (77, ROOM_TOP, 6)
        elif current_room == 7:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 6)
        elif current_room == 10:
            return (77, ROOM_TOP), (77, ROOM_BOT, 4)
        elif current_room == 4:
            return (77, ROOM_TOP), (77, ROOM_BOT, 0)
        elif current_room == 0:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 1)
        elif current_room == 11:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 12)
        elif current_room == 12:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 13)
        elif current_room == 13:
            return (77, ROOM_TOP), (77, ROOM_BOT, 7)
        
    
    if dest_room == 4:
        if current_room == 0:
            return (77, ROOM_BOT), (77, ROOM_TOP, 4)
        elif current_room == 3:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235)
        elif current_room == 5:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 4)
        elif current_room == 10:
            return (77, ROOM_TOP), (77, ROOM_BOT, 4)
        elif current_room == 1:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 0)
    
    if dest_room == 10:
        if current_room == 4:
            return (77, ROOM_BOT), (77, ROOM_TOP, 10)
        elif current_room == 0:
            return (77, ROOM_BOT), (77, ROOM_TOP, 4)
        elif current_room == 11:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 10)
        elif current_room == 6:
            return (77, ROOM_TOP), (77, ROOM_BOT, 2)
        elif current_room == 2:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 1)
        elif current_room == 1:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 0)
    
    if dest_room == 9:
        if current_room == 3:
            return (77, ROOM_BOT), (77, ROOM_TOP, 9)
        elif current_room == 4:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 3)
        elif current_room == 8:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 9)
        elif current_room == 0:
            return (77, ROOM_BOT), (77, ROOM_TOP, 4)
        elif current_room == 5:
            return (ROOM_LEFT,235), (ROOM_RIGHT, 235, 4)
        elif current_room == 10:
            return (77, ROOM_TOP), (77, ROOM_BOT, 4)
        elif current_room == 22:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 21)
        elif current_room == 21:
            return (77, ROOM_TOP), (77, ROOM_BOT, 13)
        elif current_room == 13:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 12)
        elif current_room == 12:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 11)
        elif current_room == 11:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 10)
    
    if dest_room == 11:
        if current_room == 5:
            return (77, ROOM_BOT), (77, ROOM_TOP, 11)
        elif current_room == 19:
            return (77, ROOM_TOP), (77, ROOM_BOT, 11)    
        elif current_room == 10:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 11)
        elif current_room == 12:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 11)
        elif current_room == 20:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 19)
        elif current_room == 18:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 19)
        elif current_room == 6:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 7)
        elif current_room == 7:
            return (77, ROOM_BOT), (77, ROOM_TOP, 13)
        elif current_room == 13:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 12)
    
    if dest_room == 19:
        if current_room == 5:
            return (77, ROOM_BOT), (77, ROOM_TOP, 11)
        elif current_room == 11:
            return (77, ROOM_BOT), (77, ROOM_TOP, 19)    
        elif current_room == 10:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 11)
        elif current_room == 12:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 11)
        elif current_room == 20:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 19)
        elif current_room == 18:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 19)
    
    if dest_room == 13:
        if current_room == 7:
            return (77, ROOM_BOT), (77, ROOM_TOP, 13)
        if current_room == 12:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 13)
        if current_room == 14:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 13)
        if current_room == 21:
            return (77, ROOM_TOP), (ROOM_BOT, 235, 13)
        if current_room == 22:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 21)
        if current_room == 6:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 7)
        if current_room == 19:
            return (77, ROOM_TOP), (77, ROOM_BOT, 11)
        if current_room == 11:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 12)
    
    if dest_room == 21:
        if current_room == 7:
            return (77, ROOM_BOT), (77, ROOM_TOP, 13)
        if current_room == 12:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 13)
        if current_room == 14:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 13)
        if current_room == 13:
            return (77, ROOM_BOT), (ROOM_TOP, 235, 21)
        if current_room == 22:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 21)
    
    if dest_room == 22:
        if current_room == 14:
            return (77, ROOM_BOT), (77, ROOM_TOP, 22)
        if current_room == 21:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 22)
        if current_room == 23:
            return (ROOM_LEFT, 235), (ROOM_RIGHT, 235, 22)
        if current_room == 9:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 10)
        if current_room == 10:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 11)
        if current_room == 11:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 12)
        if current_room == 12:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 13)
        if current_room == 13:
            return (ROOM_RIGHT, 235), (ROOM_LEFT, 235, 14)
    
    print("Current room: {} dest room: {} not configured".format(current_room, dest_room))

def get_dist_from_term(term, true_terms):
    
    dists = []
    
    def manhatten(point1,point2):
        distance = abs(point1[0]-point2[0]) + abs(point1[1]-point2[1])
        return distance
    
    for true_term in true_terms:
        dist = 0
        
        while term[2] != true_term[2]:
            dest, new_term = get_dist_point(term[2], true_term[2])
            dist += manhatten(dest, term)
            term = new_term
        
        dist += manhatten(term, true_term)
        dists.append(dist)
    
    return min(dists)

def get_dists(df,
                        seen_rooms,
                        seeds,
                        env_idxs,
                        head_idxs,
                        env_idx_to_room_num):
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
                        (df['num_rooms'] == num_rooms)&
                        (df['seed'] == seed)&
                        (df['env_idx'] == env_idx)&
                        (df['head_idx'] == head_idx)
                    ]
                    if len(mini_df) == 0:
                        head_results.append(10000)
                    else:
                        terms = mini_df.iloc[-100:]['final_location']
                        dists = []
                        for term in terms:
                           dists.append(get_dist_from_term(term, TERMS))
                        head_results.append(np.mean(dists))
                env_results.append(min(head_results))
            seed_results.append(np.mean(env_results))
        plot_points_avg[num_rooms-1] = np.mean(seed_results)
        plot_points_var[num_rooms-1] = np.std(seed_results) 
    
    return plot_points_avg, plot_points_var
        
def term_dist_by_seen_plot(df, env_idx_to_room_idx, ax, line_label):
    head_idxs = df['head_idx'].unique()
    env_idxs = df['env_idx'].unique()
    rooms = df['num_rooms'].unique()
    seeds = df['seed'].unique()
    avg, std = get_dists(df,
                                  rooms,
                                  seeds,
                                  env_idxs,
                                  head_idxs,
                                  env_idx_to_room_idx)
    
    print("=========================")
    print(avg)
    print(std)
    print("=========================")
    
    ax.plot(avg, rooms, label=line_label)

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

def room_success_by_seen_plot(df, fig_name):
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
    
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot()
    ax.plot(avg)
    fig.savefig(fig_name)
    
    
    
def room_scatter_plots(df, plot_folder):
    head_idxs = df['head_idx'].unique()
    env_idxs = df['env_idx'].unique()
    rooms = df['num_rooms'].unique()
    seeds = df['seed'].unique()
    
    os.makedirs(plot_folder, exist_ok=True)
    
    for head_idx in head_idxs:
        for env_idx in env_idxs:
            for room in rooms:
                for seed in seeds:
                    mini_df = df.loc[
                        (df['num_rooms']==room)&
                        (df['seed']==seed)&
                        (df['env_idx']==env_idx)&
                        (df['head_idx']==head_idx)
                    ]
                    if len(mini_df) > 0:
                        term_points = mini_df.iloc[-100:]['final_location']
                        scatter_folder = os.path.join(plot_folder, "head{}_envidx{}_numroom{}_seed{}".format(head_idx,
                                                                                                             env_idx,
                                                                                                             room,
                                                                                                             seed))
                        scatter_from_terms(term_points, scatter_folder)

def joined_room_scatter(df, 
                        plot_folder, 
                        seed,
                        env_idx,
                        head_idxs,
                        room_num):
    room_plots = {}
    markers = [".","X"]
    
    for head_idx in head_idxs:
        mini_df = df.loc[
            (df['num_rooms']==room_num)&
            (df['seed']==seed)&
            (df['env_idx']==env_idx)&
            (df['head_idx']==head_idx)
        ]
        if len(mini_df) > 0:
            term_points = mini_df.iloc[-100:]['final_location']
            
            room_x = defaultdict(list)
            room_y = defaultdict(list)
            
            for loc in term_points:
                room_x[loc[2]].append(loc[0])
                room_y[loc[2]].append(loc[1])
            
            for room in room_x.keys():
                if room not in room_plots:
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.set_ylim([0,300])
                    ax.set_xlim([0,160])
                    room_plots[room] = (fig, ax)
                
                room_plots[room][1].scatter(room_x[room], room_y[room])
    
    scatter_folder = os.path.join(plot_folder, "envidx{}_numrooms{}_seed{}".format(env_idx,
                                                                                   room_num,
                                                                                   seed))
    for room in room_plots.keys():
        file_name = os.path.join(scatter_folder, "heads_{}_{}".format(head_idxs[0], head_idxs[1]))
        room_plots[room][1].savefig(file_name)
            


file_dir = "/mnt/nfs/home/ademello/storage/portable_options/new_runs/"

files = get_data_files(file_dir, "ladder")
rooms, seeds = get_rooms_seeds(files, 8)
df = get_combined_df(files, rooms, seeds)
print(df)
# room_success_by_seen_plot(df, "runs/ladder.png")
# room_scatter_plots(df, "runs/scatter_ladder")
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()
term_dist_by_seen_plot(df, [1,0,2,3,5,7,14], ax, "D-BAT Ensemble")
fig.savefig("runs/ladder_dist.png")



