import os 
import gin
import torch 
import numpy as np 
import torch.optim as optim
from copy import deepcopy
from pfrl import explorers 
from pfrl import replay_buffers
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils.batch_states import batch_states
import torch.nn as nn

from portable.option.policy.agents import Agent
from portable.option.policy.models import LinearQFunction, compute_q_learning_loss
from portable.option.divdis.policy.models.cnn_q_function import CNNQFunction
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
                 learn_initiation,
                 max_len_init_classifier=500,
                 max_len_context_classifier=500,
                 steps_to_bootstrap_init_classifier=1000,
                 q_hidden_size=64,
                 image_input=True,
                 discount_rate=0.9,
                 initiation_maxlen=100):
        super().__init__()

        self.use_gpu = use_gpu 
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
        
        self.q_network = LinearQFunction(in_features=gru_hidden_size,
                                         n_actions=num_actions,
                                         hidden_size=q_hidden_size)
        
        self.recurrent_memory = nn.GRU(input_size=policy_infeature_size,
                                       hidden_size=gru_hidden_size,
                                       batch_first=True)
        
        self.target_q_network = deepcopy(self.q_network)
        self.target_q_network.eval()
        
        self.policy_optimizer = optim.Adam(list(self.q_network.parameters()) \
            + list(self.recurrent_memory.parameters())\
                + list(self.cnn.parameters()),
                                           lr=learning_rate)
        
        # classifier to determine if in initiation classifier
        self.initiation = FactoredInitiationClassifier(maxlen=max_len_init_classifier)
        # classifier to determine if state is part of existing context
        self.context = FactoredContextClassifier(maxlen=max_len_context_classifier)
        
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
    
    def can_initiate(self, obs):
        in_context = self.context.predict(obs)
        print(self.initiation.pessimistic_predict(obs))
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
    
    def update_step(self):
        self.step_number += 1
    
    def observe(self,
                obs,
                action,
                reward,
                next_obs,
                terminal):
        self.update_step()
        
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs)
        
        if type(next_obs) == np.ndarray:
            next_obs = torch.from_numpy(next_obs)
        
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
            self.replay_updater.update_if_necessary(self.step_number)
    
    def update(self, experiences, errors_out=None):
        if self.training:
            has_weight = "weight" in experiences[0][0]
            if self.use_gpu:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
            exp_batch = batch_experiences(
                experiences,
                device=device,
                phi=self.phi,
                gamma=self.gamma,
                batch_states=batch_states
            )
            # get weights for prioritized experience replay
            if has_weight:
                exp_batch["weights"] = torch.tensor(
                    [elem[0]["weight"] for elem in experiences],
                    device=device,
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
        if self.use_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        obs = batch_states([obs], device, self.phi)
        if self.image_input:
            obs = self.cnn(obs)
        obs = obs.unsqueeze(1).float()
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
        if self.use_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        obs = batch_states(obs, device, self.phi)
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
        self.replay_buffer.save(os.path.join(dir, 'buffer.pkl'))
        np.save(os.path.join(dir, "step_number.npy"), self.step_number)
    
    def load(self, dir):
        if os.path.exists(os.path.join(dir, "policy.pt")):
            print("\033[92m {}\033[00m" .format("Policy model loaded"))
            self.q_network.load_state_dict(torch.load(os.path.join(dir, 'policy.pt')))
            if self.image_input:
                self.cnn.load_state_dict(torch.load(os.path.join(dir, 'cnn.pt')))
            self.target_q_network.load_state_dict(torch.load(os.path.join(dir, 'policy.pt')))
            self.recurrent_memory.load_state_dict(torch.load(os.path.join(dir, 'recurrent_mem.pt')))
            self.replay_buffer.load(os.path.join(dir, 'buffer.pkl'))
            self.step_number = np.load(os.path.join(dir, "step_number.npy"))
        else:
            print("\033[91m {}\033[00m" .format("No Checkpoint found. No model has been loaded"))
    


