from copy import deepcopy
from portable.option.policy.models import compute_q_learning_loss
import torch
import torch.optim as optim
import numpy as np 

class OptionAgentModel():
    def __init__(self,
                 action_agent,
                 option_agent,
                 use_gpu,
                 learning_rate,
                 gamma,
                 num_actions,
                 
                 summary_writer=None,
                 video_generator=None) -> None:
        
        self.action_agent = action_agent
        self.option_agent = option_agent
        self.use_gpu = use_gpu
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_actions = num_actions
        
        self.summary_writer = summary_writer
        self.video_generator = video_generator
        
        self.target_action_agent = deepcopy(action_agent)
        self.target_action_agent.eval()
        self.target_option_agent = deepcopy(option_agent)
        self.target_option_agent.eval()
        
        # self.action_optimizer = optim.Adam(
        #     self.action_agent.parameters(),
        #     lr=learning_rate
        # )
        self.option_optimizer = optim.Adam(
            self.option_agent.parameters(),
            lr=learning_rate
        )
        
        self.timestep = 0
        
        if self.use_gpu:
            self.option_agent.to("cuda")
            self.target_option_agent.to("cuda")
        
    def save(self, save_path):
        self.option_agent.save(save_path)
        self.action_agent.save(save_path)
    
    def load(self, load_path):
        self.option_agent.load(load_path)
        self.action_agent.load(load_path)
    
    def step(self,
             state,
             action,
             reward,
             next_state,
             done,
             reset):
        
        self.action_agent.step(state,
                               action,
                               reward,
                               next_state,
                               done, 
                               reset)
    
    def train(self,
              exp_batch,
              errors_out=None,
              update_target_network=False):
        
        """
        Train option agent.
        """
        
        self.option_agent.train()
        
        batch_states = exp_batch['state']
        batch_actions = exp_batch['action']
        batch_rewards = exp_batch['reward']
        batch_next_states = exp_batch['next_state']
        batch_dones = exp_batch['is_state_terminal']
        batch_options = exp_batch['option']
        
        # train action model
        # batch_pred_q_values_action = self.action_agent(batch_states)
        # batch_pred_values_action = batch_pred_q_values_action.gather(dim=1, index=batch_actions)
        
        batch_next_state_q_values_action = self.target_action_agent.q_function(batch_next_states)
        batch_next_state_values_action = torch.argmax(batch_next_state_q_values_action, axis=1)
        # batch_q_target_action = batch_rewards + self.gamma*(1-batch_dones)*batch_next_state_values_action
        
        # td_loss_action = compute_q_learning_loss(exp_batch,
        #                                          batch_pred_values_action,
        #                                          batch_q_target_action,
        #                                          errors_out=errors_out)
        
        # loss_action = td_loss_action.item()
        
        # self.action_optimizer.zero_grad()
        # td_loss_action.backward()
        # self.action_optimizer.step()
        
        # train option model
        action_vectors = np.zeros((len(batch_states), self.num_actions))
        action_vectors[batch_actions.cpu().numpy()] = 1
        
        batch_pred_q_values_option = self.option_agent(action_vectors, batch_states)
        batch_options = torch.unsqueeze(batch_options, dim=0)
        
        batch_pred_values_option = batch_pred_q_values_option.gather(dim=1, index=batch_options)
        
        next_action_vectors = np.zeros((len(batch_next_states), self.num_actions))
        next_action_vectors[batch_next_state_values_action.cpu().numpy()] = 1
        next_action_vectors = torch.from_numpy(next_action_vectors).to(batch_next_states.device)
        
        batch_next_state_q_values_option = self.target_option_agent(next_action_vectors, batch_next_states)
        batch_next_state_values_option = torch.argmax(batch_next_state_q_values_option, axis=1)
        batch_q_target_option = batch_rewards + self.gamma*(1-batch_dones)*batch_next_state_values_option
        
        td_loss_option = compute_q_learning_loss(exp_batch,
                                                 batch_pred_values_option.squeeze(),
                                                 batch_q_target_option.squeeze(),
                                                 errors_out=errors_out)
        
        loss_option = td_loss_option.item()
        
        self.option_optimizer.zero_grad()
        td_loss_option.backward()
        self.option_optimizer.step()
        
        if update_target_network:
            self.target_action_agent = deepcopy(self.action_agent)
            self.target_action_agent.eval()
            self.target_option_agent.load_state_dict(self.option_agent.state_dict())
        
        if self.summary_writer is not None:
            # self.summary_writer.add_scalar('action_td_loss',
            #                                loss_action,
            #                                self.timestep)
            self.summary_writer.add_scalar('option_td_loss',
                                           loss_option,
                                           self.timestep)
        
        self.timestep += 1
    
    def _video_log(self, line):
        if self.video_generator is not None:
            self.video_generator.add_line(line)
    
    def predict_action(self, 
                       state,
                       action_mask,
                       option_mask,
                       epsilon):
        """
        given a state, return a predicted action and option
        """
        with torch.no_grad():
            self.action_agent.batch_act([state])
            action_q_values = self.action_agent.q_function(state)[0]
            action_q_values[np.logical_not(action_mask)] = -1e8
            
            self._video_log("[option agent] action q values: {}".format(action_q_values))
            
            rand_val = np.random.rand()
            
            if rand_val < epsilon:
                self._video_log("[option agent] random select")
                action = np.random.randint(0, len(action_q_values))
                while not action_mask[action]:
                    action = np.random.randint(0, len(action_q_values))
            else:
                self._video_log("[option agent] greedy select")
                action = torch.argmax(action_q_values)
            
            self._video_log("[option agent] selected action: {}".format(action))
            
            action_vector = np.zeros(self.num_actions)
            action_vector[action] = 1
            option_q_values = self.option_agent([action_vector], state)[0]
            chosen_option_mask = option_mask[action]
            
            if np.sum(chosen_option_mask) == 0:
                return action, 0
            
            chosen_option_mask = chosen_option_mask.astype(bool)
            
            
            option_q_values[not chosen_option_mask] = -1e8
            self._video_log("[option agent] option q values: {}".format(option_q_values))
            
            if rand_val < epsilon:
                option = np.random.randint(0, len(option_q_values))
                while not chosen_option_mask[option]:
                    option = np.random.randint(0, len(option_q_values))
            else:
                option_q_values[option_mask] = -1e8
                option = torch.argmax(option_q_values)
            
            self._video_log("[option agent] chosen option: {}".format(option))
        
        return action, option