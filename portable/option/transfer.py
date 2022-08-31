import os
import csv
import argparse
from pathlib import Path

import pfrl
import numpy as np
from matplotlib import pyplot as plt

from portable import utils
from portable.option.option_utils import SingleOptionTrial
from portable.option.train import train_ensemble_agent_with_eval
from portable.option.policy import EnsembleAgent, DoubleDQNAgent


class TransferTrial(SingleOptionTrial):
    """
    load a trained agent, and try to retrain it on another starting spot
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
        parser.add_argument("--load", "-l", type=str, required=True,
                            help="the experiment_name of the trained agent so we know where to look for loading it")
        parser.add_argument("--target", "-t", type=str, required=True, 
                            nargs='+', default=[],
                            help="a list of target start_state to transfer to")
        parser.add_argument("--plot", "-p", action='store_true',
                            help="only do the plotting. Use this after the agent has been trained on transfer tasks.")
        parser.add_argument("--agent", type=str, choices=['dqn', 'ensemble'], default='ensemble',
                            help="the type of agent to transfer on")
        parser.add_argument("--num_policies", type=int, default=3,
                            help="number of policies in the ensemble")
        
        # testing params
        parser.add_argument("--steps", type=int, default=50000,
                            help="max number of steps to train the agent for")

        args = self.parse_common_args(parser)
        return args

    def _set_experiment_name(self):
        """
        the experiment name shall the the combination of the loading state as well as all the transfer targets
        """
        connector = '->'
        exp_name = self.params['load']
        for target in self.params['target']:
            exp_name += connector + target
        self.params['experiment_name'] = exp_name
    
    def check_params_validity(self):
        super().check_params_validity()
        # check that all the target start_states are valid
        for target in self.params['target']:
            self.find_start_state_ram_file(target)
        print(f"Targetting {len(self.params['target'])} transfer targets: {self.params['target']}")
        # set experiment name
        self._set_experiment_name()
        # log more frequently because it takes less time to train
        self.params['reward_logging_freq'] = 100
        self.params['success_rate_save_freq'] = 1  # in episodes
        self.params['eval_freq'] = 500
        self.params['saving_freq'] = self.params['steps']

    def setup(self):
        super().setup()
        self.check_params_validity()
        
        # setting random seeds
        pfrl.utils.set_random_seed(self.params['seed'])

        # find loading dir
        self.loading_dir = Path(self.params['results_dir']) / self.params['load'] / self.expanded_agent_name
        try: 
            assert self.loading_dir.exists()
        except AssertionError:
            # for termination-clf agents etc, there is no such dir in the pre-training dir
            self.loading_dir = Path(self.params['results_dir']) / self.params['load'] / self.detailed_agent_name

        # get the hyperparams
        hyperparams_file = self.loading_dir / 'hyperparams.csv'
        self.saved_params = utils.load_hyperparams(hyperparams_file)

        # create the saving directories
        self.saving_dir = self._set_saving_dir()
        if not self.params['plot']:
            utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['saving_dir'] = self.saving_dir

        # save the hyperparams
        if not self.params['plot']:
            utils.save_hyperparams(os.path.join(self.saving_dir, "hyperparams.csv"), self.params)

    def plot_results(self):
        """
        just plot the results after the agent has been trained on transfer tasks
        """
        plotting_dir = Path(self.params['results_dir']) / self.params['experiment_name']
        plot_when_well_trained(self.params['target'], plotting_dir)
        plot_average_success_rate(self.params['target'], plotting_dir)
    
    def transfer(self):
        """
        sequentially train the agent on each of the targets
        loaded agent -> first target state
        first target state trained -> second target state
        second target state trained -> third target state
        ...
        """
        # training
        trained = self.params['load']
        for i, target in enumerate(self.params['target']):
            print(f"Training {trained} -> {target}")
            # make env
            env = self.make_env(self.saved_params['environment'], self.saved_params['seed'], eval=False, start_state=target)
            eval_env = self.make_env(self.saved_params['environment'], self.saved_params['seed']+1000, eval=True, start_state=target)
            # find loaded agent
            if trained == self.params['load']:
                agent_file = self.loading_dir / 'agent.pkl'
            else:
                agent_file = sub_saving_dir / 'agent.pkl'
            # make saving dir
            exp_name = trained + '->' + target
            sub_saving_dir = self.saving_dir.joinpath(exp_name)
            sub_saving_dir.mkdir()
            # make agent
            plots_dir = sub_saving_dir / 'plots'
            plots_dir.mkdir()
            if self.params['agent'] == 'dqn':
                agent = DoubleDQNAgent.load(agent_file)
            elif self.params['agent'] == 'ensemble':
                agent = EnsembleAgent.load(agent_file, reset=True, plot_dir=plots_dir)
            # train
            train_ensemble_agent_with_eval(
                agent,
                env,
                max_steps=self.params['steps'],
                saving_dir=sub_saving_dir,
                success_rate_save_freq=self.params['success_rate_save_freq'],
                reward_save_freq=self.params['reward_logging_freq'],
                agent_save_freq=self.params['saving_freq'],
                eval_env=eval_env,
                eval_freq=self.params['eval_freq'],
                success_threshold_for_well_trained=self.params['success_threshold_for_well_trained'],
                success_queue_size=self.params['success_queue_size'],
            )
            # advance to next target
            trained = target
        
        # meta learning statistics
        self.plot_results()
    
    def run(self):
        if self.params['plot']:
            self.plot_results()
        else:
            self.transfer()


def _rotate_xticks():
    """
    rotate the xticks and make it a small font so that they don't overlap each other
    """
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')


def _grab_when_well_trained_data(targets, dir):
    """
    given a dir, find all the well_trained csv files and put the data into two arrays
    """
    steps_when_well_trained = np.zeros(len(targets))
    episode_when_well_trained = np.zeros(len(targets))

    # descend into the sub saving dirs to find the well_trained csv file
    for subdir in os.listdir(dir):
        if not os.path.isdir(dir / subdir):
            continue
        well_trained_file = dir / subdir / 'eval_well_trained_time.csv'
        try:
            with open(well_trained_file, 'r') as f:
                csv_reader = csv.reader(f)
                data = list(csv_reader)[-1]
                episode = int(data[0])
                step = int(data[1])
        except FileNotFoundError:
            # the file is not logged because the agent was not well trained.
            step = np.inf
            episode = np.inf
        target = subdir.split('->')[1]
        steps_when_well_trained[targets.index(target)] = step
        episode_when_well_trained[targets.index(target)] = episode
    
    return steps_when_well_trained, episode_when_well_trained


def plot_when_well_trained(targets, saving_dir):
    """
    given the saving_dir/results_dir, find all the agent_dir within it and plot the 
    when_well_trained data on one graph, comparing all the agents
    """
    # grab data
    agent_to_data = {}
    for agent_dir in os.listdir(saving_dir):
        if not os.path.isdir(saving_dir / agent_dir):
            continue
        steps_when_well_trained, episode_when_well_trained = _grab_when_well_trained_data(targets, saving_dir / agent_dir)
        agent_to_data[agent_dir] = {
            'steps': steps_when_well_trained,
            'episodes': episode_when_well_trained,
        }
    
    # plot steps when well trained
    steps_file = saving_dir / 'steps_when_well_trained.png'
    for agent in agent_to_data:
        plt.plot(agent_to_data[agent]['steps'], label=agent)
    plt.xticks(range(len(targets)), targets)
    plt.xlabel('target')
    plt.ylabel('steps till skill is well trained')
    plt.legend()
    _rotate_xticks()
    plt.savefig(steps_file)
    plt.close()

    # plot episodes when well trained
    episode_file = saving_dir / 'episode_when_well_trained.png'
    for agent in agent_to_data:
        plt.plot(agent_to_data[agent]['episodes'], label=agent)
    plt.xticks(range(len(targets)), targets)
    plt.xlabel('target')
    plt.ylabel('episode till skill is well trained')
    plt.legend()
    _rotate_xticks()
    plt.savefig(episode_file)
    plt.close()


def _grab_average_success_rate_data(targets, dir):
    """
    given a dir, find all the average_success_rate csv files and put the data into two arrays
    """
    average_success_rates = np.zeros(len(targets))
    # descend into the sub saving dirs to find the success rates file
    for subdir in os.listdir(dir):
        if not os.path.isdir(dir.joinpath(subdir)):
            continue
        success_rates_file = Path(dir) / subdir / 'eval_success_rate.csv'
        with open(success_rates_file, 'r') as f:
            csv_reader = csv.reader(f)
            success_rates = [float(row[1]) for row in csv_reader]
            avg_success_rate = np.mean(success_rates)
            target = subdir.split('->')[-1]
            average_success_rates[targets.index(target)] = avg_success_rate
    return average_success_rates


def plot_average_success_rate(targets, saving_dir):
    # grab data
    agent_to_data = {}
    for agent_dir in os.listdir(saving_dir):
        if not os.path.isdir(saving_dir / agent_dir):
            continue
        average_success_rates = _grab_average_success_rate_data(targets, saving_dir / agent_dir)
        agent_to_data[agent_dir] = average_success_rates

    # plot
    for agent in agent_to_data:
        plt.plot(agent_to_data[agent], label=agent)
    plt.xticks(range(len(targets)), targets)
    plt.xlabel('target')
    plt.ylabel('average success rate')
    plt.legend()
    _rotate_xticks()
    plt.savefig(saving_dir / 'average_success_rate.png')
    plt.close()


def main():
    trial = TransferTrial()
    trial.run()


if __name__ == '__main__':
    main()
