from portable.agent.bonus_based_exploration.run_experiment import create_exploration_agent, create_exploration_runner
import numpy as np
import os
import logging
import datetime
import lzma
import dill
from portable.option.sets.utils import get_vote_function, VOTE_FUNCTION_NAMES
import torch
import random
from portable.option import Option
import pandas as pd
import gin
import matplotlib.pyplot as plt

@gin.configurable
class RainbowExperiment():

    def __init__(self,
                 base_dir,
                 seed,
                 experiment_name,
                 primitive_action_num,      # number of primitive actions
                 action_num,                # number of additional actions on top of primitive
                 starting_action_num,       # number of starting options
                 initiation_vote_function,
                 termination_vote_function,
                 policy_phi,
                 experiment_env_function,
                 make_plots=True,
                 device_type="cpu",
                 train_initiation=True,
                 options_initiation_positive_files=[[]],
                 options_initiation_negative_files=[[]],
                 options_initiation_priority_negative_files=[[]],
                 train_initiation_embedding_epochs=50,
                 train_initiation_classifier_epochs=50,
                 train_termination=True,
                 options_termination_positive_files=[[]],
                 options_termination_negative_files=[[]],
                 options_termination_priority_negative_files=[[]],
                 train_termination_embedding_epochs=50,
                 train_termination_classifier_epochs=50,
                 train_policy=True,
                 policy_bootstrap_envs=[],
                 true_init_functions=[],
                 train_policy_max_steps=10000,
                 train_policy_success_rate=0.9,
                 ):
        assert device_type in ["cpu", "cuda"]
        assert initiation_vote_function in VOTE_FUNCTION_NAMES
        assert termination_vote_function in VOTE_FUNCTION_NAMES

        self.device = torch.device(device_type)
        
        random.seed(seed)
        self.seed = seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.seed))
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.plot_dir = os.path.join(self.base_dir, "plots")
        self.save_dir = os.path.join(self.base_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        self.action_num = action_num

        self.plot = make_plots

        self.x = []
        self.y = []
        self.room = []
        self.actions = []

        self.env = experiment_env_function(self.seed)
        self.eval = False

        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.INFO
        )

        self.agent = create_exploration_runner(
            os.path.join(self.base_dir, 'rainbow'),
            create_exploration_agent,
            schedule='option_agent'
        )

        self.trial_data = pd.DataFrame([],
            columns=[
                "episode_num",
                "reward",
                "steps",
                "actions_taken"
            ]
        )

        self.primitive_action_num = primitive_action_num

        self.init_vote_function = get_vote_function(initiation_vote_function)
        self.term_vote_function = get_vote_function(termination_vote_function)
        self.policy_phi = policy_phi
        self.options = []
        for idx in range(starting_action_num):
            self.options.append(
                Option(
                    device=self.device,
                    initiation_vote_function=self.init_vote_function,
                    termination_vote_function=self.term_vote_function,
                    policy_phi=self.policy_phi,
                    original_initiation_function=true_init_functions[idx]
                )
            )

        if train_initiation is True:
            self._train_initiation(
                starting_action_num,
                options_initiation_positive_files,
                options_initiation_negative_files,
                options_initiation_priority_negative_files,
                train_initiation_embedding_epochs,
                train_initiation_classifier_epochs
            )

        if train_termination is True:
            self._train_termination(
                starting_action_num,
                options_termination_negative_files,
                options_termination_positive_files,
                options_termination_priority_negative_files,
                train_termination_embedding_epochs,
                train_termination_classifier_epochs
            )

        if train_policy is True:
            self._train_policy(
                starting_action_num,
                policy_bootstrap_envs,
                train_policy_max_steps,
                train_policy_success_rate
            )

        self.save()
                

    def _train_initiation(self,
                          starting_action_num,
                          positive_files,
                          negative_files,
                          priority_negative_files,
                          embedding_epochs,
                          classifier_epochs):

        assert starting_action_num == len(negative_files)
        assert starting_action_num == len(positive_files)
        assert starting_action_num == len(priority_negative_files)
        for idx in range(starting_action_num):
            self.options[idx].initiation.add_data_from_files(
                positive_files[idx],
                negative_files[idx],
                priority_negative_files[idx]
            )
            self.options[idx].initiation.train(
                embedding_epochs,
                classifier_epochs
            )

    def _train_termination(self,
                           starting_action_num,
                           positive_files,
                           negative_files,
                           priority_negative_files,
                           embedding_epochs,
                           classifier_epochs):
        assert starting_action_num == len(negative_files)
        assert starting_action_num == len(positive_files)
        assert starting_action_num == len(priority_negative_files)
        for idx in range(starting_action_num):
            self.options[idx].termination.add_data_from_files(
                positive_files[idx],
                negative_files[idx],
                priority_negative_files[idx],
            )
            self.options[idx].termination.train(
                embedding_epochs,
                classifier_epochs
            )

    def _train_policy(self,
                      starting_action_num,
                      bootstrap_envs,
                      max_steps,
                      success_rate_for_well_trained):
        assert starting_action_num == len(bootstrap_envs)
        for idx in range(starting_action_num):
            self.options[idx].bootstrap_policy(
                bootstrap_envs[idx],
                max_steps,
                success_rate_for_well_trained
            )

    def save(self):
        base_option_path = os.path.join(self.save_dir, 'options')

        for idx in range(len(self.options)):
            save_dir = os.path.join(base_option_path, str(idx))
            self.options[idx].save(save_dir)
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        with lzma.open(file_name, 'wb') as f:
            dill.dump(self.trial_data, f)

    def load(self):
        
        # gonna need to do this using some file handling
        base_option_path = os.path.join(self.save_dir, 'options')
        for idx in range(len(self.options)):
            save_dir = os.path.join(base_option_path, str(idx))
            self.options[idx].load(save_dir)
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        if os.path.exists(file_name):
            with lzma.open(file_name, 'rb') as f:
                self.trial_data = dill.load(f)


    def get_available_actions(self, info):
        # returns a mask of all available actions
        mask = np.zeros(self.action_num)
        for idx in range(self.primitive_action_num):
            mask[idx] = 1
        for idx, option in enumerate(self.options):
            global_vote, markov_vote = option.can_initiate(
                info["stacked_agent_state"],
                (info["player_x"], info["player_y"])
            )
            if global_vote is 1 or markov_vote is True:
                mask[self.primitive_action_num + idx] = 1
        
        return mask
            

    def initialize_episode(self):
        state, info = self.env.reset()
        mask = self.get_available_actions(info)
        action = self.agent.initialize_episode(info["stacked_state"].reshape((1,84,84,4)).numpy(), mask)

        self.x = []
        self.y = []
        self.room = []
        self.actions = []

        return state, info, action, mask

    def execute_action(self, action, state, info):

        self.actions.append(action)
        x, y, room = info["position"]
        self.x.append(x)
        self.y.append(y)
        self.room.append(room)

        if action < self.primitive_action_num:
            state, reward, done, info = self.env.step(action)
            steps = 1
        else:
            option_idx = action - self.primitive_action_num
            if option_idx > len(self.options):
                raise Exception('Attempted action that does not exist')
            well_trained_instances = []
            position = info["position"]
            for idx, instance in enumerate(self.options[option_idx].markov_instantiations):
                if instance.is_well_trained():
                    well_trained_instances.append(idx)
            if len(well_trained_instances) > 0 and self.options[option_idx].identify_original_initiation(position):
                logging.info("[experiment:execute_action] testing markov instance for assimilation")
                test_idx = random.choice(well_trained_instances)
                self.options[option_idx].markov_instantiations[test_idx].assimilate_run(
                    self.env,
                    state,
                    info
                )
                can_assimilate = self.options[option_idx].markov_instantiations[test_idx].can_assimilate()
                if can_assimilate is True:
                    logging.info("[experiment:execute_action] markov instance can be assimilated")
                    self.options[option_idx].update_option(
                        self.options[option_idx].markov_instantiations[test_idx],
                        1000,
                        20,
                        80
                    )
                    del self.options[option_idx].markov_instantiations[test_idx]
                elif can_assimilate is False:
                    logging.info("[experiment:execute_action] markov instance cannot be assimilated and is being created as a new option")
                    if self.primitive_action_num + len(self.options) < self.action_num:
                        new_option = Option(
                            device=self.device,
                            initiation_vote_function=self.init_vote_function,
                            termination_vote_function=self.term_vote_function,
                            policy_phi=self.policy_phi
                        )
                        new_option.original_markov_initiation = self.options[option_idx].markov_instantiations[test_idx].initiation
                        new_option.identify_original_initiation = new_option.original_markov_initiation.predict
                        new_option.update_option(
                            self.options[option_idx].markov_instantiations[test_idx],
                            1000,
                            20,
                            150
                        )
                        self.options.append(new_option)
                    del self.options[option_idx].markov_instantiations[test_idx]
            else:
                logging.info("[experiment:execute_action] running option as normal")
                state, reward, done, info, steps = self.options[option_idx].run(
                    self.env, 
                    state, 
                    info, 
                    self.eval
                )

        return state, reward, done, info, steps

    def run_episode(self, eval=False):

        self.eval = eval

        state, info, action, mask = self.initialize_episode()
        episode_reward = 0
        total_steps = 0
        actions_taken = 1


        while True:
            # print(self.agent._agent.get_q_values())
            state, reward, done, info, steps = self.execute_action(
                action,
                state,
                info
            )
            episode_reward += reward
            total_steps += steps

            if done:
                logging.info("[experiment] Episode ended.")
                self.agent.end_episode(
                    reward,
                    action,
                    mask,
                    done
                )
                break
            if info['needs_reset']:
                logging.info("[experiment] Episode timed out.")
                break

            # rainbow agent record step
            self.agent.step(reward, info["stacked_state"].reshape((1,84,84,4)).numpy(), action, mask)
            mask = self.get_available_actions(info)
            action = self.agent.select_action(mask)
            actions_taken += 1

        logging.info("[experiment] Total Reward: {} Env steps: {} Actions taken: {}".format(
            episode_reward,
            total_steps,
            actions_taken
        ))
        print("[experiment] Total Reward: {} Env steps: {} Actions taken: {}".format(
            episode_reward,
            total_steps,
            actions_taken
        ))

        self.plot_episode

        return episode_reward, total_steps, actions_taken

    def plot_episode(self):
        
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.room = np.array(self.room)
        self.actions = np.array(self.actions)

        unique_rooms = np.unique(self.room)

        for room in unique_rooms:
            room_mask = self.room == room
            fig = plt.figure(num=1, clear=True)
            ax = fig.add_subplot()
            ax.scatter(self.x[room_mask], self.y[room_mask], marker='x')

            plot_name = os.path.join(self.plot_dir, self.episode)
            os.makedirs(plot_name, exist_ok=True)
            plot_name = os.path.join(plot_name, 'room_{}.png'.format(room))
            fig.savefig(plot_name, bbox_inches='tight')
        
        unique_actions = np.unique(self.actions)

        for action in unique_actions:
            action_mask = self.actions == action
            action_rooms = self.room[action_mask]
            action_x = self.x[action_mask]
            action_y = self.y[action_mask]
            for room in np.unique(action_rooms):
                room_mask = action_rooms == room

                fig = plt.figure(num=1, clear=True)
                ax = fig.add_subplot()
                ax.scatter(action_x[room_mask], action_y[room_mask], marker='x')

                plot_name = os.path.join(self.plot_dir, self.episode, action)
                os.makedirs(plot_name, exist_ok=True)
                plot_name = os.path.join(self.plot_dir, self.episode, action, 'room_{}.png'.format(room))
                fig.savefig(plot_name, bbox_inches='tight')

    def run_trial(self, max_steps):
        total_steps = 0
        self.episode = 0
        while total_steps < max_steps:
            episode_rewards, episode_steps, episode_action_num = self.run_episode(
                eval=False
            )
            logging.info("[experiment] Episode {}".format(self.episode))
            print("[experiment] Episode {}".format(self.episode))
            d = pd.DataFrame(
                [
                    {
                        "episode_num": self.episode,
                        "reward": episode_rewards,
                        "steps": episode_steps,
                        "actions_taken": episode_action_num
                    }
                ]
            )
            self.trial_data.append(d)
            self.episode += 1

            total_steps += episode_action_num
            logging.info("Actions taken: {}".format(total_steps))
            print("Actions taken: {}".format(total_steps))
        
        self.save()



