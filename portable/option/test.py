import os
import argparse
from pathlib import Path

import pfrl
import numpy as np

from portable import utils
from portable.option.option_utils import SingleOptionTrial
from portable.option.policy import EnsembleAgent
from portable.option.policy.agents.abstract_agent import evaluating
from portable.option.policy.ensemble_utils import visualize_state_with_ensemble_actions, \
    visualize_state_with_action


def test_ensemble_agent(agent, env, saving_dir, visualize=False, num_episodes=10):
    """
    test the ensemble agent:
        - success rate
        - visualize the trajectory and keep track of epsodic reward (if visualize=True)
    """
    with evaluating(agent):
        action_meanings = env.unwrapped.get_action_meanings()
        success_rates = np.zeros(num_episodes)
        for i in range(num_episodes):
            # set random seed for each run
            pfrl.utils.set_random_seed(i+1000)
            env.seed(i+1000)
            env.action_space.seed(i+1000)

            # set up save dir
            if visualize:
                visualization_dir = os.path.join(saving_dir, f"eval_episode_{i}")
                os.mkdir(visualization_dir)

            # init
            obs = env.reset()  # reset all other wrappers
            step = 0
            total_reward = 0
            terminal = False

            while not terminal:
                # step
                if type(agent) == EnsembleAgent:
                    a, ensemble_actions, ensemble_q_vals = agent.act(obs, return_ensemble_info=True)
                else:
                    a = agent.act(obs)  # DQN
                next_obs, reward, done, info = env.step(a)
                reached_goal = info.get('reached_goal', False)
                terminal = reached_goal or info['dead'] or info['needs_reset']
                total_reward += reward

                # visualize
                if visualize:
                    save_path = os.path.join(visualization_dir, f"{step}.png")
                    if type(agent) == EnsembleAgent:
                        meaningful_actions = [action_meanings[i] for i in ensemble_actions]
                        meaningful_q_vals = [str(round(q, 2)) for q in ensemble_q_vals]
                        action_taken = str(action_meanings[a])
                        visualize_state_with_ensemble_actions(
                            obs,
                            meaningful_actions,
                            meaningful_q_vals,
                            action_taken,
                            reward,
                            save_path,
                        )
                    else:
                        # DQN
                        visualize_state_with_action(obs, str(action_meanings[a]), save_path)

                # advance
                step += 1
                obs = next_obs
            
            # epsidoe end
            success_rates[i] = 1 if reached_goal else 0
            if visualize:
                print(f"episode {i} reward: {total_reward} len: {step}")
                save_total_reward_info(total_reward, visualization_dir)
    
    # end of eval
    eval_success_rate = np.mean(success_rates)
    if visualize:
        print(f"eval success rate: {eval_success_rate}")
    return eval_success_rate


def save_total_reward_info(reward, save_dir):
    """
    save the total reawrd obtained at the end of a testing episode to disk
    """
    file = os.path.join(save_dir, "total_reward.txt")
    with open(file, "w") as f:
        f.write(str(reward))


class TestTrial(SingleOptionTrial):
    """
    load the trained agent the step through the envs to see if the Q values and the 
    action taken make sense
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
        parser.set_defaults(experiment_name="visualize")
        parser.add_argument("--load", "-l", type=str, required=True,
                            help="the experiment_name of the trained agent to load")
        
        # testing params
        parser.add_argument("--episodes", type=int, default=10,
                            help="number of episodes to test")
        parser.add_argument("--steps", type=int, default=50,
                            help="max number of steps per episode")

        args = self.parse_common_args(parser)
        return args

    def check_params_validity(self):
        return super().check_params_validity()

    def setup(self):
        self.check_params_validity()
        # setting random seeds
        pfrl.utils.set_random_seed(self.params['seed'])

        # get the hyperparams
        hyperparams_file = Path(self.params['load']).parent / 'hyperparams.csv'
        saved_params = utils.load_hyperparams(hyperparams_file)

        # create the saving directories
        self.saving_dir = Path(self.params['results_dir']).joinpath(self.params['experiment_name'])
        utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['saving_dir'] = self.saving_dir

        # env
        self.env = self.make_env(saved_params['environment'], saved_params['seed'] + 1000, eval=True, start_state=self.params['start_state'])

        # agent
        agent_file = Path(self.params['load']) / 'agent.pkl'
        self.agent = EnsembleAgent.load(agent_file)
    
    def run(self):
        """
        test the loaded agent
        """
        test_ensemble_agent(
            self.agent, 
            self.env, 
            self.saving_dir,
            visualize=True,
            num_episodes=self.params['episodes'], 
        )


def main():
    trial = TestTrial()
    trial.run()


if __name__ == '__main__':
    main()
