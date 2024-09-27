import os 
import logging
import gin
import torch 
import pickle
import numpy as np 
import torch.optim as optim
from copy import deepcopy
from pfrl import explorers 
from pfrl import replay_buffers
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils.batch_states import batch_states
import torch.nn as nn
from collections import deque
logger = logging.getLogger(__name__)

from portable.option.policy.agents import Agent
from portable.option.policy.models import LinearQFunction, compute_q_learning_loss
from portable.option.divdis.policy.models.factored_initiation_model import FactoredInitiationClassifier
from portable.option.divdis.policy.models.factored_context_model import FactoredContextClassifier

@gin.configurable
class PolicyWithInitiation(Agent):
    def __init__(self,
                 use_gpu,
                 warmup_steps,
                 prioritized_replay_anneal_steps,
                 buffer_length,
                 update_interval,
                 q_target_update_interval,
                 learning_rate,
                 final_epsilon,
                 final_exploration_frames,
                 batch_size,
                 num_actions,
                 policy_infeature_size,
                 policy_phi,
                 gru_hidden_size,
                 learn_initiation=False,
                 save_replay_buffer=False,
                 max_len_init_classifier=500,
                 max_len_context_classifier=500,
                 steps_to_bootstrap_init_classifier=1000,
                 q_hidden_size=64,
                 image_input=True,
                 discount_rate=0.99,
                 initiation_maxlen=100):
        super().__init__()

        self.device = torch.device('cuda:{}'.format(use_gpu)) 
        if use_gpu == -1:
            self.device = torch.device('cpu')
        self.warmup_steps = warmup_steps
        self.prioritized_replay_anneal_steps = prioritized_replay_anneal_steps
        self.buffer_length = buffer_length
        self.update_interval = update_interval
        self.q_target_update_interval = q_target_update_interval
        self.learning_rate = learning_rate
        self.final_epsilon = final_epsilon
        self.final_exploration_frames = final_exploration_frames
        self.gamma = discount_rate
        self.num_actions = num_actions
        self.learn_initiation = learn_initiation
        self.bootstrap_init_timesteps = steps_to_bootstrap_init_classifier
        self.interactions = 0
        
        self.step_number = 0
        self.train_rewards = deque(maxlen=200)
        self.option_runs = 0
        self.save_buffer = save_replay_buffer
        
        self.image_input = image_input
        if image_input:
            self.cnn = nn.Sequential(
                nn.LazyConv2d(out_channels=16, kernel_size=3, stride=1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                
                nn.LazyConv2d(out_channels=32, kernel_size=3, stride=1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                
                nn.Flatten()
            )
            self.cnn.to(self.device)
        
        self.q_network = LinearQFunction(in_features=gru_hidden_size,
                                         n_actions=num_actions,
                                         hidden_size=q_hidden_size)
        self.q_network.to(self.device)
        
        self.recurrent_memory = nn.GRU(input_size=policy_infeature_size,
                                       hidden_size=gru_hidden_size,
                                       batch_first=True)
        self.recurrent_memory.to(self.device)
        
        self.target_q_network = deepcopy(self.q_network)
        self.target_q_network.eval().to(self.device)
        
        self.policy_optimizer = optim.Adam(list(self.q_network.parameters()) \
            + list(self.recurrent_memory.parameters())\
                + list(self.cnn.parameters()),
                                           lr=learning_rate)
        
        # classifier to determine if in initiation classifier
        # self.initiation = FactoredInitiationClassifier(maxlen=max_len_init_classifier)
        # # classifier to determine if state is part of existing context
        # self.context = FactoredContextClassifier(maxlen=max_len_context_classifier)
        
        self.phi = policy_phi
        
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
        
        self.store_buffer_to_disk = False
        
        logger.info("Policy hps")
        logger.info("======================================")
        logger.info("======================================")
        logger.info("warmup steps: {}".format(warmup_steps))
        logger.info("learning rate: {}".format(learning_rate))
        logger.info("prioritized replay anneal: {}".format(prioritized_replay_anneal_steps))
        logger.info("buffer length: {}".format(buffer_length))
        logger.info("final epsilon: {}".format(final_epsilon))
        logger.info("final epsilon frame num: {}".format(final_exploration_frames))
        logger.info("batch size: {}".format(batch_size))
        logger.info("policy in features: {}".format(policy_infeature_size))
        logger.info("gru hidden size: {}".format(gru_hidden_size))
        logger.info("q hidden size: {}".format(q_hidden_size))
        logger.info("======================================")
        logger.info("======================================")
    
    def can_initiate(self, obs):
        in_context = self.context.predict(obs)
        if (self.interactions < self.bootstrap_init_timesteps) or (in_context is False):
            return in_context
        else:
            return self.initiation.pessimistic_predict(obs)
    
    def add_init_examples(self, 
                          positive_examples=[],
                          negative_examples=[]):
        if len(positive_examples) > 0:
            self.initiation.add_positive_examples(positive_examples)
        if len(negative_examples) > 0:
            self.initiation.add_negative_examples(negative_examples)
        
        self.initiation.fit()
        self.interactions += 1
    
    def add_context_examples(self, positive_examples):
        self.context.add_positive_examples(positive_examples)
        
        self.context.fit()
    
    def move_to_gpu(self):
        self.q_network.to("cuda")
        if self.image_input:
            self.cnn.to("cuda")
        self.target_q_network.to("cuda")
        self.recurrent_memory.to("cuda")
    
    def move_to_cpu(self):
        self.q_network.to("cpu")
        if self.image_input:
            self.cnn.to("cpu")
        self.target_q_network.to("cpu")
        self.recurrent_memory.to("cpu")
    
    def store_buffer(self, dir):
        if not self.store_buffer_to_disk:
            return
        if self.replay_buffer.memory is None:
            print("MAYBE A PROBLEM -> NO BUFFER TO SAVE")
            return
        print("storing buffer")
        os.makedirs(dir, exist_ok=True)
        self.replay_buffer.save(os.path.join(dir, 'buffer.pkl'))
        self.replay_buffer.memory = None
    
    def load_buffer(self, dir):
        if not self.store_buffer_to_disk:
            return
        print("loading buffer")
        if os.path.exists(dir):
            self.replay_buffer.load(os.path.join(dir, 'buffer.pkl'))
        else:
            print("No memory to load. Skipping")
    
    def update_step(self):
        self.step_number += 1
    
    def end_skill(self, summed_reward):
        self.train_rewards.append(summed_reward)
        self.option_runs += 1
        if self.option_runs%1 == 0:
            logger.info("Option policy success rate: {} from {} episodes {} steps".format(np.mean(self.train_rewards), 
                                                                                           self.option_runs,
                                                                                           self.step_number))
    
    def observe(self,
                obs,
                action,
                reward,
                next_obs,
                terminal):
        self.update_step()
        
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs)
        obs = obs.int()
        
        if type(next_obs) == np.ndarray:
            next_obs = torch.from_numpy(next_obs)
        next_obs = next_obs.int()
        
        if self.training:
            transition = {"state": obs,
                          "action": action,
                          "reward": reward,
                          "next_state": next_obs,
                          "next_action": None,
                          "is_state_terminal": terminal}
            self.replay_buffer.append(**transition)
            if terminal:
                self.replay_buffer.stop_current_episode()
            try:
                self.replay_updater.update_if_necessary(self.step_number)
            except:
                print("Policy update failed. Continue without train step.")
    
    def update(self, experiences, errors_out=None):
        if self.training:
            has_weight = "weight" in experiences[0][0]
            
            exp_batch = batch_experiences(
                experiences,
                device=self.device,
                phi=self.phi,
                gamma=self.gamma,
                batch_states=batch_states
            )
            # get weights for prioritized experience replay
            if has_weight:
                exp_batch["weights"] = torch.tensor(
                    [elem[0]["weight"] for elem in experiences],
                    device=self.device,
                    dtype=torch.float32
                )
                if errors_out is None:
                    errors_out = []
            
            update_target_net = self.step_number%self.q_target_update_interval==0
            self._train_policy(exp_batch, errors_out, update_target_net)
            
            if has_weight:
                assert isinstance(self.replay_buffer, replay_buffers.PrioritizedReplayBuffer)
                self.replay_buffer.update_errors(errors_out)
            
    
    def _train_policy(self,
                      exp_batch,
                      errors_out=None,
                      update_target_network=False):
        self.q_network.train()
        self.recurrent_memory.flatten_parameters()
        
        batch_obs = exp_batch['state']
        batch_actions = exp_batch['action']
        batch_rewards = exp_batch['reward']
        batch_next_obs = exp_batch['next_state']
        batch_dones = exp_batch['is_state_terminal']
        
        batch_obs = batch_obs.float()
        if self.image_input:
            batch_obs = self.cnn(batch_obs)
        batch_obs = batch_obs.unsqueeze(1)
        batch_obs, _ = self.recurrent_memory(batch_obs)
        batch_obs = batch_obs.squeeze()
        batch_pred_q_all_actions = self.q_network(batch_obs)
        batch_pred_q = batch_pred_q_all_actions.evaluate_actions(batch_actions)
        
        with torch.no_grad():
            batch_next_obs = batch_next_obs.float()
            if self.image_input:
                batch_next_obs = self.cnn(batch_next_obs)
            batch_next_obs = batch_next_obs.unsqueeze(1)
            batch_next_obs, _ = self.recurrent_memory(batch_next_obs)
            batch_next_obs = batch_next_obs.squeeze()
            batch_next_pred_q_all_actions = self.target_q_network(batch_next_obs)
            next_state_values = batch_next_pred_q_all_actions.max
            batch_q_target = batch_rewards + self.gamma*(1-batch_dones)*next_state_values
        
        loss = compute_q_learning_loss(exp_batch,
                                       batch_pred_q,
                                       batch_q_target,
                                       errors_out=errors_out)
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        if update_target_network:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, obs, return_q=False):
                
        obs = batch_states([obs], self.device, self.phi)
        obs = obs.float()
        if self.image_input:
            obs = self.cnn(obs)
        obs = obs.unsqueeze(1)
        obs, _ = self.recurrent_memory(obs)
        obs = obs.squeeze(0)
        q_values = self.q_network(obs)
        
        if self.training:
            a = self.explorer.select_action(
                self.step_number,
                greedy_action_func=lambda: q_values.greedy_actions
            )
        else:
            randval = np.random.rand()
            if randval > 0.01:
                a = q_values.greedy_actions
            else:
                a = np.random.randint(0, self.num_actions)
        if return_q is True:
            return a, q_values.q_values
        return a

    def batch_act(self, obs):
        obs = batch_states(obs, self.device, self.phi)
        if self.image_input:
            obs = self.cnn(obs)
        obs = obs.unsqueeze(1).float()
        obs, _ = self.recurrent_memory(obs)
        obs = obs.squeeze(0)
        q_values = self.q_network(obs)

        randval = np.random.rand()
        if randval > 0.01:
            a = q_values.greedy_actions
        else:
            a = np.random.randint(0, self.num_actions)
        
        return a, q_values.q_values
    
    def add_data_initiation(self,
                            positive_examples=[],
                            negative_examples=[]):
        if len(positive_examples) != 0:
            self.initiation.add_positive_examples(positive_examples)
        if len(negative_examples) != 0:
            self.initiation.add_negative_examples(negative_examples)
        
        self.initiation.fit()
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        
        torch.save(self.q_network.state_dict(), os.path.join(dir, 'policy.pt'))
        if self.image_input:
            torch.save(self.cnn.state_dict(), os.path.join(dir, 'cnn.pt'))
        torch.save(self.recurrent_memory.state_dict(), os.path.join(dir, 'recurrent_mem.pt'))
        if self.save_buffer is True:
            if self.store_buffer_to_disk is False:
                self.replay_buffer.save(os.path.join(dir, 'buffer.pkl'))
        np.save(os.path.join(dir, "step_number.npy"), self.step_number)
        np.save(os.path.join(dir, "option_runs.npy"), self.option_runs)
        with open(os.path.join(dir, "run_rewards.pkl"), "wb") as f:
            pickle.dump(self.train_rewards, f)
    
    def load(self, dir):
        if os.path.exists(os.path.join(dir, "policy.pt")):
            print("\033[92m {}\033[00m" .format("Policy model loaded"))
            self.q_network.load_state_dict(torch.load(os.path.join(dir, 'policy.pt')))
            if self.image_input:
                self.cnn.load_state_dict(torch.load(os.path.join(dir, 'cnn.pt')))
            self.target_q_network.load_state_dict(torch.load(os.path.join(dir, 'policy.pt')))
            self.recurrent_memory.load_state_dict(torch.load(os.path.join(dir, 'recurrent_mem.pt')))
            if self.save_buffer is True:
                if self.store_buffer_to_disk is False:
                    self.replay_buffer.load(os.path.join(dir, 'buffer.pkl'))
            self.step_number = np.load(os.path.join(dir, "step_number.npy"))
            self.option_runs = np.load(os.path.join(dir, "option_runs.npy"))
            with open(os.path.join(dir, "run_rewards.pkl"), "rb") as f:
                self.train_rewards = pickle.load(f)
        else:
            print("\033[91m {}\033[00m" .format("No Checkpoint found. No model has been loaded"))
    

