import gin 
from pfrl import explorers
from pfrl import replay_buffers
from pfrl.replay_buffer import ReplayUpdater
from pfrl.utils.batch_states import batch_states
from pfrl.collections.prioritized import PrioritizedBuffer
import numpy as np 
import math
import torch

from portable.agent.model.option_agent_model import OptionAgentModel

@gin.configurable
class OptionAgent():
    
    def __init__(self,
                 action_agent,
                 option_agent,
                 use_gpu,
                 phi,
                 num_actions,
                 batch_size,
                 warmup_steps, # maybe not needed
                 buffer_length,
                 update_interval,
                 q_target_update_interval,
                 learning_rate,
                 final_epsilon,
                 final_exploration_frames,
                 discount_rate,
                 prioritized_replay_anneal_steps,
                 summary_writer=None,
                 video_generator=None):
        self.agent = OptionAgentModel(action_agent=action_agent,
                                      option_agent=option_agent,
                                      use_gpu=use_gpu,
                                      learning_rate=learning_rate,
                                      gamma=discount_rate,
                                      num_actions=num_actions,
                                      summary_writer=summary_writer,
                                      video_generator=video_generator)
        self.use_gpu = use_gpu
        self.phi = phi
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.buffer_length = buffer_length
        self.update_interval = update_interval
        self.q_target_update_interval = q_target_update_interval
        self.learning_rate = learning_rate
        self.final_epsilon = final_epsilon
        self.final_exploration_frames = final_exploration_frames
        self.discount_rate = discount_rate
        self.num_actions = num_actions
        self.video_generator = video_generator
        
        self.summary_writer = summary_writer
        
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            final_epsilon,
            final_exploration_frames,
            lambda: np.random.randint(num_actions)
        )
        
        self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            capacity=buffer_length,
            alpha=0.5,
            beta0=0.4,
            betasteps=prioritized_replay_anneal_steps,
            normalize_by_max="memory"
        )
        
        self.replay_updater = ReplayUpdater(
            replay_buffer=self.replay_buffer,
            update_func=self.update,
            batchsize=batch_size,
            episodic_update=False,
            episodic_update_len=None,
            n_times_update=1,
            replay_start_size=warmup_steps,
            update_interval=update_interval
        )
        
        self.is_training = True
        self.step_number = 0
        
        self._cumulative_discount_vector = np.array(
            [math.pow(discount_rate, n) for n in range(100)]
        )
    
    @staticmethod
    def batch_experiences(experiences, device, phi, gamma, batch_states=batch_states):
        """Takes a batch of k experiences each of which contains j

        consecutive transitions and vectorizes them, where j is between 1 and n.

        Args:
            experiences: list of experiences. Each experience is a list
                containing between 1 and n dicts containing
                - state (object): State
                - action (object): Action
                - reward (float): Reward
                - is_state_terminal (bool): True iff next state is terminal
                - next_state (object): Next state
            device : GPU or CPU the tensor should be placed on
            phi : Preprocessing function
            gamma: discount factor
            batch_states: function that converts a list to a batch
        Returns:
            dict of batched transitions
        """
        
        batch_exp = {
            "state": batch_states([elem[0]["state"] for elem in experiences], device, phi),
            "action": torch.as_tensor(
                [elem[0]["action"] for elem in experiences], device=device
            ),
            "option": torch.as_tensor(
                [elem[0]["option"] for elem in experiences], device=device
            ),
            "reward": torch.as_tensor(
                [
                    sum((gamma ** i) * exp[i]["reward"] for i in range(len(exp)))
                    for exp in experiences
                ],
                dtype=torch.float32,
                device=device,
            ),
            "next_state": batch_states(
                [elem[-1]["next_state"] for elem in experiences], device, phi
            ),
            "is_state_terminal": torch.as_tensor(
                [
                    any(transition["is_state_terminal"] for transition in exp)
                    for exp in experiences
                ],
                dtype=torch.float32,
                device=device,
            ),
            "discount": torch.as_tensor(
                [(gamma ** len(elem)) for elem in experiences],
                dtype=torch.float32,
                device=device,
            ),
        }
        if all(elem[-1]["next_action"] is not None for elem in experiences):
            batch_exp["next_action"] = torch.as_tensor(
                [elem[-1]["next_action"] for elem in experiences], device=device
            )
        return batch_exp
    
    def update(self, experiences, errors_out=None):
        
        if self.is_training:
            has_weight = "weight" in experiences[0][0]
            if self.use_gpu:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            exp_batch = self.batch_experiences(experiences=experiences,
                                               device=device,
                                               phi=self.phi,
                                               gamma=self.discount_rate,
                                               batch_states=batch_states)
            # get weights for prioritized experience replay
            if has_weight:
                exp_batch["weights"] = torch.tensor(
                    [elem[0]["weight"] for elem in experiences],
                    device=device,
                    dtype=torch.float32
                )
                if errors_out is None:
                    errors_out = []
            
            update_target_net = self.step_number % self.q_target_update_interval == 0
            self.agent.train(exp_batch, errors_out, update_target_net)
            if has_weight:
                assert isinstance(self.replay_buffer, replay_buffers.PrioritizedReplayBuffer)
                self.replay_buffer.update_errors(errors_out)
    
    def save(self, save_path):
        pass
    
    def load(self, load_path):
        pass
    
    def observe(self,
                obs,
                action,
                option_idx,
                rewards,
                next_obs,
                terminal):
        
        self.step_number += 1
        
        if len(rewards) > len(self._cumulative_discount_vector):
            self._cumulative_discount_vector = np.array(
                [math.pow(self.discount_rate, n) for n in range(len(rewards))]
            )
        
        reward = np.sum(self._cumulative_discount_vector[:len(rewards)]*rewards)
        
        
        if self.is_training:
            transition = {
                "state": obs,
                "action": action,
                "reward": reward,
                "next_state": next_obs,
                "next_action": None,
                "is_state_terminal": terminal,
                "option": option_idx
            }
            
            self.replay_buffer.append(**transition)
            if terminal:
                self.replay_buffer.stop_current_episode()
            
            self.replay_updater.update_if_necessary(self.step_number)
            
            self.agent.step(obs,
                            action,
                            reward,
                            next_obs,
                            terminal,
                            False)
            
    
    def compute_epsilon(self):
        if self.is_training is False:
            return 0.01
        if self.step_number > self.final_exploration_frames:
            return self.final_epsilon
        else:
            epsilon_diff = self.final_epsilon - 1
            return 1 - epsilon_diff * (self.step_number/self.final_exploration_frames)
    
    def act(self,
            obs,
            action_mask,
            option_mask):
        if self.use_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        obs = batch_states([obs], device, self.phi)
        action, option = self.agent.predict_action(obs,
                                                                     action_mask,
                                                                     option_mask,
                                                                     self.compute_epsilon())
        
        return action, option
