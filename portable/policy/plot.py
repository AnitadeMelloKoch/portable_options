import os
import pickle

import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def plot_reward_curve(csv_dir):
    """
    this is used to plot for a single agent
    read progress.csv and plot the reward curves, save in the save dir as csv
    """
    csv_path = os.path.join(csv_dir, 'progress.csv')
    df = pandas.read_csv(csv_path, comment='#')

    # get rid of the NaN data points
    max_nan_step = df.loc[df.isna().any(axis=1)]['level_total_steps'].max()
    df = df.query(f"level_total_steps > {max_nan_step}")

    steps = df['total_steps']
    train_reward = df['ep_reward_mean']
    eval_reward = df['eval_ep_reward_mean']
    plt.plot(steps, train_reward, label='train')
    plt.plot(steps, eval_reward, label='eval')
    plt.legend()
    plt.title('Learning Curve')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.dirname(csv_path) + '/learning_curve.png'
    plt.savefig(save_path)
    plt.close()


def pretty_title(game_name):
    """
    make the first letter upper case
    """
    return game_name[0].upper() + game_name[1:]


def plot_eight_procgen_games(results_dir):
    """
    plot the eight procgen games in one big plot
    """
    games = ['bigfish', 'coinrun', 'dodgeball', 'heist', 'jumper', 'leaper', 'maze', 'ninja']
    fig, axes = plt.subplots(2, 4, sharex=True)
    for i, game in enumerate(games):
        experiment_dir = os.path.join(results_dir, game)
        # get data
        rewards_mean = process_training_curve_csv_file(experiment_dir)
        # plot
        sns.lineplot(
            ax=axes[i // 4, i % 4],
            data=rewards_mean,
            x='level_total_steps',
            y='ep_reward_mean',
            hue='agent',
            style='agent',
        )
        # title 
        axes[i // 4, i % 4].set_title(pretty_title(game))
        # ylabel
        if i % 4 == 0:
            axes[i // 4, i % 4].set_ylabel('Episodic Reward')
        else:
            axes[i // 4, i % 4].set_ylabel('')
        # xlabel
        axes[i // 4, i % 4].set_xlabel('Steps')
        # shared legend
        axes[i // 4, i % 4].legend().remove()
        if i == 7:
            handles, labels = axes[i // 4, i % 4].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=4)
            plt.subplots_adjust(bottom=0.15)

    # save
    save_path = os.path.join(results_dir, 'procgen_results.png')
    fig.savefig(save_path)
    with open(os.path.join(results_dir, 'procgen_results.pkl'), 'wb') as f:
        pickle.dump(fig, f)
    print(f'saved to {save_path}')


def process_training_curve_csv_file(exp_dir):
    """
    read from the progress.csv file and return a dataframe with the relevant information
    find all the csv files in exp_dir (all seeds, and all agents) and process all
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        agent_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')
            assert df['total_steps'].max() == 10_000_000, "total steps is not complete (20 * 500k)"  # check that csv is complete
            df = df[['level_total_steps', 'level_index', 'ep_reward_mean']].copy()
            df['agent'] = agent
            df['seed'] = int(seed)
            rewards.append(df)
    rewards = pandas.concat(rewards, ignore_index=True)
    max_nan_step = rewards.loc[rewards.isna().any(axis=1)]['level_total_steps'].max()
    subset = rewards.query(f"level_total_steps > {max_nan_step}")
    # average across different level_index
    rewards_mean = subset.groupby(['level_total_steps', 'agent', 'seed']).mean().reset_index()

    return rewards_mean

def plot_transfer_exp_training_curve_across_levels(exp_dir):
    """
    x-axis: steps in each level
    y-axis: reward, averaged across different levels
    """
    rewards_mean = process_training_curve_csv_file(exp_dir)
    # plot
    sns.lineplot(
        data=rewards_mean,
        x='level_total_steps',
        y='ep_reward_mean',
        hue='agent',
        style='agent',
    )
    plt.title(f'Training Curve Averaged Across Levels :{exp_dir}')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.dirname(exp_dir) + '/training_curve.png'
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()


def plot_transfer_exp_eval_curve(exp_dir):
    """
    x-axis: levels
    y-axis: eval reward at that level, averaged across the last few timesteps
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        agent_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')
            df = df[['level_total_steps', 'eval_ep_reward_mean', 'level_index']].copy()
            df = df.groupby('level_index').tail(100)  # only keep the last 20 timesteps
            df = df.groupby('level_index').mean().reset_index()  # and mean across those timesteps
            df['agent'] = agent
            df['seed'] = int(seed)
            rewards.append(df)
    rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='level_index',
        y='eval_ep_reward_mean',
        hue='agent',
        style='agent',
    )
    plt.title(f'Eval Reward after Trained on Level 1 - k: {exp_dir}')
    plt.xlabel('Level')
    plt.ylabel('Eval Reward (averaged over last 20 steps at level k)')
    plt.xticks(range(len(rewards['level_index'].unique())))
    save_path = os.path.dirname(exp_dir) + '/eval_curve.png'
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()


def plot_reward_curve(csv_dir):
    """
    this is used to plot for a single agent
    read progress.csv and plot the reward curves, save in the save dir as csv
    """
    csv_path = os.path.join(csv_dir, 'progress.csv')
    df = pandas.read_csv(csv_path, comment='#')

    # get rid of the NaN data points
    max_nan_step = df.loc[df.isna().any(axis=1)]['level_total_steps'].max()
    df = df.query(f"level_total_steps > {max_nan_step}")

    steps = df['total_steps']
    train_reward = df['ep_reward_mean']
    eval_reward = df['eval_ep_reward_mean']
    plt.plot(steps, train_reward, label='train')
    plt.plot(steps, eval_reward, label='eval')
    plt.legend()
    plt.title('Learning Curve')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.dirname(csv_path) + '/learning_curve.png'
    plt.savefig(save_path)
    plt.close()


def plot_train_eval_curve(exp_dir, kind='eval'):
    """
    plot the eval-curve of ensemble 1 and ensemble 3 and compare
    """
    assert kind in ['eval', 'train']
    keyword = 'eval_ep_reward_mean' if kind == 'eval' else 'ep_reward_mean'
    rewards = []
    for agent in os.listdir(exp_dir):
        if agent not in ['ensemble-1', 'ensemble-3']:
            continue
        agent_dir = os.path.join(exp_dir, agent)
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            
            df = pandas.read_csv(csv_path, comment='#')
            # get rid of the NaN data points
            max_nan_step = df.loc[df.isna().any(axis=1)]['level_total_steps'].max()
            df = df.query(f"level_total_steps > {max_nan_step}")

            df = df[['total_steps', keyword]].copy()
            sparsity = 5  # only plot every 4 points
            df = df[df.total_steps % (sparsity * 800) == 0]
            df[[keyword]] = df[[keyword]].rolling(20).mean()  # rolling mean to denoise
            df['agent'] = agent
            df['seed'] = int(seed)
            rewards.append(df)

    rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='total_steps',
        y=keyword,
        hue='agent',
        style='agent'
    )
    plt.title(f'{kind} Curve')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.dirname(exp_dir) + f'/{kind}_curve.png'
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()


def plot_all_agents_reward_data(exp_dir):
    """
    given an experiments dir, find all the subdirs that represent different agents
    and gather their eval_ep_reward_mean data
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        agent_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')
            # get rid of the NaN data points
            max_nan_step = df.loc[df.isna().any(axis=1)]['level_total_steps'].max()
            df = df.query(f"level_total_steps > {max_nan_step}")
            # df = df[df['total_steps'] % 32000 == 0]

            eval_df = df[['total_steps', 'eval_ep_reward_mean']].copy()
            eval_df['seed'] = int(seed)
            eval_df['agent'] = agent
            eval_df['kind'] = 'eval'
            eval_df.rename(columns={'eval_ep_reward_mean': 'reward'}, copy=False, inplace=True)

            train_df = df[['total_steps', 'ep_reward_mean']].copy()
            train_df['seed'] = int(seed)
            train_df['agent'] = agent
            train_df['kind'] = 'train'
            train_df.rename(columns={'ep_reward_mean': 'reward'}, copy=False, inplace=True)

            new_df = pandas.concat([eval_df, train_df], ignore_index=True)
            rewards.append(new_df)
    rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='total_steps',
        y='reward',
        hue='agent',
        style='kind',
    )
    plt.title(f'Learning Curve: {exp_dir}')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.join(exp_dir, 'learning_curve.png')
    plt.savefig(save_path)
    plt.close()


def plot_all_agents_generalization_gap(exp_dir):
    """
    given an experiment dir, find all the subdirs that represent different agents
    and plot the difference between the training reward curve and the eval reward curve
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        agent_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')
            # get rid of the NaN data points
            max_nan_step = df.loc[df.isna().any(axis=1)]['level_total_steps'].max()
            df = df.query(f"level_total_steps > {max_nan_step}")

            new_df = df[['total_steps']].copy()
            new_df['seed'] = int(seed)
            new_df['agent'] = agent
            new_df['reward_diff'] = df['ep_reward_mean'] - df['eval_ep_reward_mean']
            rewards.append(new_df)
        rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='total_steps',
        y='reward_diff',
        hue='agent',
        style='agent',
    )
    plt.title(f'Generalization Gap: {exp_dir}')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Training Reward - Episodic Eval Reward')
    save_path = os.path.join(exp_dir, 'generalization_gap.png')
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', required=True, help='path to the csv file')
    parser.add_argument('--compare', '-c', action='store_true', help='compare all agents in the same dir', default=False)
    parser.add_argument('--gap', '-g', action='store_true', help='plot the generalization gap', default=False)
    parser.add_argument('--evaluation', '-e', action='store_true', help='plot the evaluation curve', default=False)
    parser.add_argument('--train', '-t', action='store_true', help='plot the training curve', default=False)
    parser.add_argument('--transfer', '-f', action='store_true', help='plot the transfer curve', default=False)
    parser.add_argument('--procgen', '-p', action='store_true', help='plot the 8 procgen games combined', default=False)
    args = parser.parse_args()
    if args.compare:
        plot_all_agents_reward_data(args.load)
    elif args.gap:
        plot_all_agents_generalization_gap(args.load)
    elif args.evaluation:
        plot_train_eval_curve(args.load, kind='eval')
    elif args.train:
        plot_train_eval_curve(args.load, kind='train')
    elif args.transfer:
        plot_transfer_exp_eval_curve(args.load)
        plot_transfer_exp_training_curve_across_levels(args.load)
    elif args.procgen:
        plot_eight_procgen_games(args.load)
    else:
        plot_reward_curve(args.load)
