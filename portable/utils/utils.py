import os
import csv
import sys
import shutil
import logging
import datetime
import argparse
from pathlib import Path
from pydoc import locate
from collections import defaultdict
from distutils.util import strtobool

import gin
import gym
import dill
import torch
import pfrl
import matplotlib.pyplot as plt
from pfrl.utils import set_random_seed
import numpy as np


def create_log_dir(dir_path, remove_existing=True, log_git=True):
    """
    Prepare a directory for outputting training results.
    Then the following infomation is saved into the directory:
        command.txt: command itself
        start_time.txt: start time of the running script
    Additionally, if the current directory is under git control, the following
    information is saved:
        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff HEAD`
    """
    outdir = os.path.join(os.getcwd(), dir_path)
    # remove existing dir
    if remove_existing:
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
            print(f"Removed existing directory {outdir}")
    # create log dir
    try:
        os.makedirs(outdir, exist_ok=not remove_existing)
    except OSError:
        print(f"Creation of the directory {outdir} failed")
    else:
        print(f"Successfully created the directory {outdir}")

    # log the command used
    with open(os.path.join(outdir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    # log the start time
    with open(os.path.join(outdir, "start_time.txt"), "w") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(timestamp)

    # log git stuff
    if log_git:
        from pfrl.experiments.prepare_output_dir import is_under_git_control, save_git_information
        if is_under_git_control():
            save_git_information(outdir)
        
    return outdir

def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)

def load_init_states(filenames):
    states = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            states.append(dill.load(f))
    return states

def load_hyperparams(filepath):
    params = dict()
    with open(filepath, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for name, value, dtype in reader:
            if dtype == 'bool':
                params[name] = bool(strtobool(value))
            else:
                params[name] = locate(dtype)(value)
    return params

def plot_state(state, plot_name):
    fig = plt.figure(num=1, clear=True)
    gs = fig.add_gridspec(nrows=1, ncols=4)

    for x in range(4):
        ax = fig.add_subplot(gs[0,x])
        ax.imshow(state[x], cmap='gray')
        ax.axis('off')

    fig.savefig(plot_name, bbox_inches='tight')
    plt.close('all')

def save_hyperparams(filepath, params):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name, value in sorted(params.items()):
            type_str = defaultdict(lambda: None, {
                bool: 'bool',
                int: 'int',
                str: 'str',
                float: 'float',
            })[type(value)] # yapf: disable
            if type_str is not None:
                writer.writerow((name, value, type_str))


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def update_param(params, name, value):
    if name not in params:
        raise KeyError(
            "Parameter '{}' specified, but not found in hyperparams file.".format(name))
    else:
        logging.info("Updating parameter '{}' to {}".format(name, value))
    if type(params[name]) == bool:
        params[name] = bool(strtobool(value))
    else:
        params[name] = type(params[name])(value)

def plot_attentions(attentions, votes, class_conf, weightings, plot_name):
    for x in range(len(attentions)):
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        ax.axis('off')
        ax.set_title('Vote: {} \n Classifier Confidence: {} \n Weight: {}'.format(
            votes[x], class_conf[x], weightings[x]
        ))
        fig_name = os.path.join(plot_name, 'attention_{}.png'.format(x))

        layer_viz = attentions[x][0,:,:,:].detach().cpu().numpy()
        mean_att = np.mean(layer_viz, axis=0)
        ax.imshow(mean_att, cmap='jet')

        fig.savefig(fig_name, bbox_inches='tight')

class BaseTrial:
    """
    a base class for running experiments
    """
    def __init__(self):
        pass

    def get_common_arg_parser(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
        )
        # common args
        # system 
        parser.add_argument("--experiment_name", "-e", type=str,
                            help="Experiment Name, also used as the directory name to save results")
        parser.add_argument("--results_dir", type=str, default='results',
                            help='the name of the directory used to store results')
        parser.add_argument("--device", type=str, default='cuda',
                            help="cpu/cuda/cuda:0/cuda:1")
        # environments
        parser.add_argument("--environment", type=str,
                            help="name of the gym environment")
        parser.add_argument("--seed", type=int, default=0,
                            help="Random seed")
        # hyperparams
        parser.add_argument('--params_dir', type=Path, default='hyperparams',
                            help='the hyperparams directory')
        parser.add_argument('--hyperparams', type=str, default='atari',
                            help='which hyperparams csv file to use')
        return parser

    def parse_common_args(self, parser):
        args, unknown = parser.parse_known_args()
        other_args = {
            (remove_prefix(key, '--'), val)
            for (key, val) in zip(unknown[::2], unknown[1::2])
        }
        args.other_args = other_args
        return args

    def load_hyperparams(self, args):
        """
        load the hyper params from args to a params dictionary
        """
        params = load_hyperparams(args.params_dir.joinpath(args.hyperparams + '.csv'))
        for arg_name, arg_value in vars(args).items():
            # if arg_name == 'hyperparams':
            #     continue
            params[arg_name] = arg_value
        for arg_name, arg_value in args.other_args:
            update_param(params, arg_name, arg_value)
        return params
    
    def make_deterministic(self, seed, force_deterministic=False):
        set_random_seed(seed)
        torch.backends.cudnn.benchmark = False  # make alg selection determistics, but may be slower
        if force_deterministic:
            torch.use_deterministic_algorithms(True)  # must use deterministic algorithms

    def make_env(self, env_name, env_seed):
        if self.params['use_deepmind_wrappers']:
            env = pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60)  # 30 min with 60 fps
            env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
                env,
                episode_life=True,
                clip_rewards=True,
                frame_stack=True,
                scale=False,
                fire_reset=True,
                channel_order="chw",
                flicker=False,
            )
        else:
            env = gym.make(env_name)
        print(f'making environment {env_name}')
        env.seed(env_seed)
        env.action_space.seed(env_seed)
        return env