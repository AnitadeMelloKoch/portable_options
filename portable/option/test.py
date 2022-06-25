import os
import argparse
from pathlib import Path

import pfrl

from portable import utils
from portable.option.option_utils import SingleOptionTrial
from portable.option.policy import EnsembleAgent
from portable.option.policy.agents.abstract_agent import evaluating
from portable.option.policy.ensemble_utils import visualize_state_with_ensemble_actions, \
    visualize_state_with_action


def test_ensemble_agent(agent, env, saving_dir, num_episodes=10, max_steps_per_episode=50):
    """
    test the ensemble agent manually by running the agent on a specific environment
    for a number of episodes
    visualize the trajectory and also keep track of the total reward
    """
    with evaluating(agent):
        action_meanings = env.unwrapped.get_action_meanings()
        for i in range(num_episodes):
            # set random seed for each run
            env.seed(i+1000)
            env.action_space.seed(i+1000)

            # set up save dir
            visualization_dir = os.path.join(saving_dir, f"trained_agent_episode_{i}")
            os.mkdir(visualization_dir)

            # init
            env.unwrapped.reset()  # real reset, or else EpisodicLife just takes Noop
            obs = env.reset()  # get the warped frame 
            step = 0
            total_reward = 0
            done = False

            while not done and step < max_steps_per_episode:
                # step
                if type(agent) == EnsembleAgent:
                    a, ensemble_actions, ensemble_q_vals = agent.act(obs, return_ensemble_info=True)
                else:
                    a = agent.act(obs)  # DQN
                next_obs, reward, done, info = env.step(a)
                total_reward += reward

                # visualize
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
                        save_path,
                    )
                else:
                    # DQN
                    visualize_state_with_action(obs, str(action_meanings[a]), save_path)

                # advance
                step += 1
                obs = next_obs
            print(f"episode {i} reward: {total_reward}")
            save_total_reward_info(total_reward, visualization_dir)


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

    def setup(self):
        # setting random seeds
        pfrl.utils.set_random_seed(self.params['seed'])

        # get the hyperparams
        hyperparams_file = Path(self.params['results_dir']) / self.params['load'] / 'hyperparams.csv'
        saved_params = utils.load_hyperparams(hyperparams_file)

        # create the saving directories
        self.saving_dir = Path(self.params['results_dir']).joinpath(self.params['experiment_name'])
        utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['saving_dir'] = self.saving_dir

        # env
        self.env = self.make_env(saved_params['environment'], saved_params['seed'] + 1000, self.params['start_state'])

        # agent
        agent_file = Path(self.params['results_dir']) / self.params['load'] / 'agent.pkl'
        self.agent = EnsembleAgent.load(agent_file)
    
    def run(self):
        """
        test the loaded agent
        """
        test_ensemble_agent(self.agent, self.env, self.saving_dir, 
                            num_episodes=self.params['episodes'], 
                            max_steps_per_episode=self.params['steps'])


def main():
    trial = TestTrial()
    trial.run()


if __name__ == '__main__':
    main()
