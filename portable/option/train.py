import time
import random
import os
import csv
import argparse
from collections import deque

import torch
import seeding
import numpy as np
import pfrl
from matplotlib import pyplot as plt

from portable import utils 
from portable.option.policy import EnsembleAgent, make_dqn_agent
from portable.option.test import test_ensemble_agent
from portable.option.option_utils import SingleOptionTrial


def train_ensemble_agent_with_eval(
    agent, 
    env, 
    max_steps, 
    saving_dir, 
    success_rate_save_freq, 
    reward_save_freq, 
    agent_save_freq,
    eval_env=None,
    eval_freq=None,
    success_queue_size=10,
    success_threshold_for_well_trained=0.9
):
    """
    run the actual experiment to train one option, and periodically evaluate
    """
    start_time = time.time()
    assert (eval_env is None) == (eval_freq is None)

    # train loop
    step_number = 0
    episode_number = 0
    total_reward = 0
    success_rates = deque(maxlen=success_queue_size)
    eval_success_rates = deque(maxlen=success_queue_size)
    step_when_well_trained = None
    episode_when_well_trained = None
    step_when_eval_well_trained = None
    episode_when_eval_well_trained = None
    state = env.reset()
    while step_number < max_steps:
        # action selection: epsilon greedy
        action = agent.act(state)
        
        # step
        next_state, reward, done, info = env.step(action)
        agent.observe(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        # log training success rate
        if done or info['needs_reset']:
            # success rate
            success_rates.append(done and not info['dead'])
            if (episode_number + 1) % success_rate_save_freq == 0:
                save_success_rate(success_rates, episode_number, saving_dir, eval=False)
            # if well trained
            well_trained = len(success_rates) >= success_queue_size/2 and get_success_rate(success_rates) >= success_threshold_for_well_trained
            if step_when_well_trained is None and well_trained:
                save_is_well_trained(saving_dir, step_number, episode_number, file_name='training_well_trained_time.csv')
                step_when_well_trained, episode_when_well_trained = step_number, episode_number
            episode_number += 1
            state = env.reset()

        # periodically eval
        if eval_freq and (step_number+1) % eval_freq == 0:
            # success rates
            eval_success = test_ensemble_agent(agent, eval_env, saving_dir, visualize=False, num_episodes=5)
            eval_success_rates.append(eval_success)
            save_success_rate([eval_success], episode_number, saving_dir, eval=True)
            # if well trained 
            eval_well_trained = len(eval_success_rates) >= success_queue_size/2 and get_success_rate(eval_success_rates) >= success_threshold_for_well_trained
            if step_when_eval_well_trained is None and eval_well_trained:
                save_is_well_trained(saving_dir, step_number, episode_number, file_name='eval_well_trained_time.csv')
                step_when_eval_well_trained, episode_when_eval_well_trained = step_number, episode_number
        
        save_total_reward(total_reward, step_number, saving_dir, reward_save_freq)
        # save_agent(agent, step_number, saving_dir, agent_save_freq)

        # advance to next
        step_number += 1

    # at the end of training
    plot_success_rate(saving_dir, eval=True)
    plot_success_rate(saving_dir, eval=False)
    plot_total_reward(saving_dir)
    save_agent(agent, step_number, saving_dir, saving_freq=1)  # always save agent at end
    test_ensemble_agent(agent, eval_env, saving_dir, visualize=True, num_episodes=1)

    end_time = time.time()

    print("Time taken: ", end_time - start_time)

    return step_when_well_trained, episode_when_well_trained


def is_well_trained(success_rates, success_threshold_for_stopping):
    """
    currently not used because prioritized replay need a fixed step number to anneal 
    beta to 1.

    determine if the agent is well trained enough for the current particular skill
    args:
        success_threshold_for_stopping: float between 0 and 1, the threshold for success rate to stop training
    """
    return get_success_rate(success_rates) >= success_threshold_for_stopping


def get_success_rate(success_rates):
    """
    args:
        success_rates: a seq of success rates
    """
    return np.mean(success_rates)


def save_success_rate(success_rates, episode_number, saving_dir, eval=False):
    """
    log the average success rate during training/testing
    the success rate at every episode is the average success rate over the last ? episodes
    """
    name = 'eval' if eval else 'training'
    save_file = os.path.join(saving_dir, f"{name}_success_rate.csv")

    # write to csv
    open_mode = 'w' if episode_number == 0 else 'a'
    with open(save_file, open_mode) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([episode_number, get_success_rate(success_rates)])


def plot_success_rate(saving_dir, eval=False):
    """plot the saved success rate"""
    name = 'eval' if eval else 'training'
    save_file = os.path.join(saving_dir, f"{name}_success_rate.csv")
    img_file = os.path.join(saving_dir, f"{name}_success_rate.png")
    with open(save_file, 'r') as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader])
        episodes = data[:, 0].astype(int)
        rates = data[:, 1].astype(np.float32)
        plt.plot(episodes, rates)
        plt.title(f"{name} success rate")
        plt.xticks(episodes)
        plt.xlabel("Episode")
        plt.ylabel("Success rate")
        plt.savefig(img_file)
        plt.close()


def save_total_reward(total_reward, step_number, saving_dir, save_every=50):
    """
    log the total reward achieved during training every 50 steps
    """
    save_file = os.path.join(saving_dir, "total_reward.csv")
    img_file = os.path.join(saving_dir, "total_reward.png")
    if step_number % save_every == 0:
        # write to csv
        open_mode = 'w' if step_number == 0 else 'a'
        with open(save_file, open_mode) as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([step_number, total_reward])


def plot_total_reward(saving_dir):
    """plot the saved total reward"""
    save_file = os.path.join(saving_dir, "total_reward.csv")
    img_file = os.path.join(saving_dir, "total_reward.png")
    with open(save_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = np.array([row for row in csv_reader])  # (step_number, 2)
        steps = data[:, 0].astype(int)
        total_reward = data[:, 1].astype(np.float32)
        plt.plot(steps, total_reward)
        plt.title("training reward")
        plt.xlabel("steps")
        plt.ylabel("total reward")
        plt.savefig(img_file)
        plt.close()


def save_is_well_trained(saving_dir, steps, episode, file_name):
    """
    save the time when training is finised: the success rate is above some threshold 
    """
    print(f"well trained at episode {episode} and step {steps}")
    time_file = os.path.join(saving_dir, file_name)
    with open(time_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["episode", "step"])
        csv_writer.writerow([episode, steps])


def save_agent(agent, step_number, saving_dir, saving_freq):
    """
    save the trained model
    """
    if (step_number+1) % saving_freq == 0:
        agent.save(saving_dir)
        print(f"model saved at step {step_number}")


class TrainEnsembleOfSkills(SingleOptionTrial):
    """
    a class for running experiments to train an option
    """
    def __init__(self):
        super().__init__()
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        self.setup()

    def parse_args(self):
        """
        parse the inputted argument
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[self.get_common_arg_parser()]
        )
        # agent
        parser.add_argument("--agent", type=str, choices=['dqn', 'ensemble'], default='ensemble',
                            help="the type of agent to train")

        # ensemble
        parser.add_argument("--num_policies", type=int, default=3,
                            help="number of policies in the ensemble")
        
        # training
        parser.add_argument("--steps", type=int, default=500000,
                            help="number of training steps")
        
        parser.add_argument("--verbose", action="store_true", default=False,
                            help="whether to print the training loss")
        args = self.parse_common_args(parser)
        return args

    def check_params_validity(self):
        """
        check whether the params entered by the user is valid
        """
        if self.params['agent'] == 'ensemble':
            try:
                assert self.params['target_update_interval'] == self.params['ensemble_target_update_interval'] * self.params['update_interval']
            except AssertionError:
                new_interval = self.params['ensemble_target_update_interval'] * self.params['update_interval']
                print(f"updating target_update_interval to be {new_interval}")
                self.params['target_update_interval'] = new_interval
    
    def setup(self):
        """
        do set up for the experiment
        """
        super().setup()
        self.check_params_validity()

        # setting random seeds
        seeding.seed(self.params['seed'], random, np)
        pfrl.utils.set_random_seed(self.params['seed'])

        # torch benchmark
        torch.backends.cudnn.benchmark = True

        # create the saving directories
        self.saving_dir = self._set_saving_dir()
        utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['saving_dir'] = self.saving_dir
        self.params['plots_dir'] = os.path.join(self.saving_dir, 'plots')
        os.mkdir(self.params['plots_dir'])

        # save the hyperparams
        utils.save_hyperparams(os.path.join(self.saving_dir, "hyperparams.csv"), self.params)

        # set up env
        self.env = self.make_env(self.params['environment'], self.params['seed'], eval=False, start_state=self.params['start_state'])
        self.eval_env = self.make_env(self.params['environment'], self.params['seed']+1000, eval=True, start_state=self.params['start_state'])

        # set up agent
        def phi(x):  # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255
        if self.params['agent'] == 'dqn':
            # DQN
            self.agent = make_dqn_agent(
                q_agent_type="DoubleDQN",
                arch="nature",
                phi=phi,
                n_actions=self.env.action_space.n,
                replay_start_size=self.params['warmup_steps'],
                buffer_length=self.params['buffer_length'],
                update_interval=self.params['update_interval'],
                target_update_interval=self.params['target_update_interval'],
            )
        else:
            self.agent = EnsembleAgent(
                device=self.params['device'],
                phi=phi,
                action_selection_strategy=self.params['action_selection_strat'],
                warmup_steps=self.params['warmup_steps'],
                batch_size=self.params['batch_size'],
                prioritized_replay_anneal_steps=self.params['steps'] / self.params['update_interval'],
                buffer_length=self.params['buffer_length'],
                update_interval=self.params['update_interval'],
                q_target_update_interval=self.params['target_update_interval'],
                num_modules=self.params['num_policies'],
                num_output_classes=self.env.action_space.n,
                plot_dir=self.params['plots_dir'],
                verbose=self.params['verbose']
            )
    
    def train_option(self):
        train_ensemble_agent_with_eval(
            self.agent,
            self.env,
            max_steps=self.params['steps'],
            saving_dir=self.saving_dir,
            success_rate_save_freq=self.params['success_rate_save_freq'],
            reward_save_freq=self.params['reward_logging_freq'],
            agent_save_freq=self.params['saving_freq'],
            eval_env=self.eval_env,
            eval_freq=self.params['eval_freq'],
            success_threshold_for_well_trained=self.params['success_threshold_for_well_trained'],
            success_queue_size=self.params['success_queue_size']
        )


def main():
    trial = TrainEnsembleOfSkills()
    trial.train_option()


if __name__ == "__main__":
    main()
