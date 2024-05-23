import numpy as np
import pandas as pd
import pickle

ep_files = ["runs/oscar/10/checkpoints/episode_results.pkl"]
ex_files = ["runs/oscar/10/checkpoints/experiment_results.pkl"]

for ex_file, ep_file in zip(ex_files, ep_files):
    with open(ex_file, 'rb') as f:
        experiment = pickle.load(f)
    with open(ep_file, 'rb') as f:
        episode = pickle.load(f)


    ex_df= pd.DataFrame.from_records(experiment, index="frames").reset_index(drop=False)
    # print(ex_df)

    ep_df = pd.DataFrame.from_records(episode, index='frames')
    # print(ep_df)
    
    new_df = []
    
    start_frame = 0
    for ep_frame, ep_data in ep_df.iterrows():
        _ex_data = ex_df[ex_df.frames > start_frame]
        _ex_data = _ex_data[_ex_data.frames <= ep_frame]
        
        episode_rewards = []
        
        for _, ex_data in _ex_data.iterrows():
            episode_rewards.extend(ex_data.option_rewards)
        
        new_row = {
            "frames": ep_frame,
            "episode": ep_data.episode,
            "episode_rewards": episode_rewards
        }
        start_frame = ep_frame
        new_df.append(new_row)

    data_df = pd.DataFrame.from_records(new_df)
    # print(data_df)
    data_df["episode_reward"] = data_df.episode_rewards.apply(np.sum)
    # # print(ep_df)
    
    data_df["rolling_reward"] = data_df.episode_reward.rolling(window=100).mean().replace(np.nan, 0)
    print(data_df)
    
    import seaborn
    plt = seaborn.lineplot(data=data_df, x="frames", y="rolling_reward")
    plt.get_figure().savefig("episode_reward.png")
    # data = []

    # for episode in experiment.options[4].train_data[4]:
    #     data.append(
    #         {
    #             "frames": episode["frames"],
    #             "reward": sum(episode["option_rewards"])
    #         }
    #     )

    # option_df = pd.DataFrame.from_records(data)
    # print(option_df)