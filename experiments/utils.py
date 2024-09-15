from portable.utils.ale_utils import get_object_position, get_skull_position
import numpy as np
import torch 
from collections import deque
import logging
import os
import datetime
from portable.utils.utils import load_gin_configs

def get_snake_x_left(position):
    if position[2] == 9:
        return 11
    if position[2] == 11:
        return 35
    if position[2] == 22:
        return 25

def get_snake_x_right(position):
    if position[2] == 9:
        return 60
    if position[2] == 11:
        return 118
    if position[2] == 22:
        return 25



def in_epsilon_square(current_position, final_position):
    epsilon = 2
    if current_position[0] <= (final_position[0] + epsilon) and \
        current_position[0] >= (final_position[0] - epsilon) and \
        current_position[1] <= (final_position[1] + epsilon) and \
        current_position[1] >= (final_position[1] - epsilon):
        return True
    return False   

def get_percent_completed_enemy(start_pos, final_pos, terminations, env):
    def manhatten(a,b):
        return sum(abs(val1-val2) for val1, val2 in zip((a[0], a[1]),(b[0],b[1])))

    if start_pos[2] != final_pos[2]:
        return 0

    info = env.get_current_info({})
    if info["dead"]:
        return 0

    room = start_pos[2]
    ground_y = start_pos[1]
    ram = env.unwrapped.ale.getRAM()
    true_distance = 0
    completed_distance = 0
    if final_pos[2] != room:
        return 0
    if final_pos[2] == 4 and final_pos[0] == 5:
        return 0
    if room in [2,3]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-25, ground_y)
        if final_pos[0] < skull_x-25 and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    if room in [0,1,18]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-6, ground_y)
        if final_pos[0] < skull_x-6 and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        end_pos = (spider_x - 6, ground_y)
        if final_pos[0] < spider_x and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    elif room in [9,11,22]:
        # snakes
        end_pos = terminations
        if in_epsilon_square(final_pos, end_pos):
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    else:
        return 0

    return completed_distance/(true_distance+1e-5)

def check_termination_correct_enemy_left(state, env):
    info = env.get_current_info({})
    if info["dead"]:
        return False
    
    position = info["position"]

    room = position[2]
    ram = env.unwrapped.ale.getRAM()
    if room in [2,3]:
        # dancing skulls
        skull_x = get_skull_position(ram)
        if position[0] < skull_x-25 and position[1] <= 235:
            return True
        else:
            return False
    if room in [1,5,18]:
        # rolling skulls
        skull_x = get_skull_position(ram)
        if room == 1:
            ground_y = 148
        elif room == 5:
            ground_y = 195
        elif room == 18:
            ground_y = 235
        if position[0] < skull_x-6 and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        ground_y = 235
        if position[0] < spider_x and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [9,11,22]:
        # snakes
        snake_x = get_snake_x_left(position)
        ground_y = 235
        if position[0] < snake_x and position[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False

def check_termination_correct_enemy_right(state, env):
    info = env.get_current_info({})
    if info["dead"]:
        return False
    
    position = info["position"]
    room = position[2]
    ram = env.unwrapped.ale.getRAM()
    if room in [2,3]:
        # dancing skulls
        skull_x = get_skull_position(ram)
        if position[0] > skull_x+25 and position[1] <= 235:
            return True
        else:
            return False
    if room in [1,5,18]:
        # rolling skulls
        skull_x = get_skull_position(ram)
        if room == 1:
            ground_y = 148
        elif room == 5:
            ground_y = 195
        elif room == 18:
            ground_y = 235
        if position[0] > skull_x+6 and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        ground_y = 235
        if position[0] > spider_x and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [9,11,22]:
        # snakes
        snake_x = get_snake_x_right(position)
        ground_y = 235
        if position[0] > snake_x and position[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False

def epsilon_ball_from_list(current_position, term_positions):
    epsilon = 4
    for term_position in term_positions:
        if current_position[0] <= (term_position[0] + epsilon) and \
            current_position[0] >= (term_position[0] - epsilon) and \
            current_position[1] <= (term_position[1] + epsilon) and \
            current_position[1] >= (term_position[1] - epsilon):
            return 1
    return 0 

def train_head(head_idx, 
                conn,
                max_steps,
                option_timeout,
                make_env,
                option,
                seed,
                learn_new_policy,
                env_idx,
                log_dir,
                init_states,
                term_states,
                config_file,
                gin_bindings):
    env = make_env(seed, init_states)
    total_steps = 0
    rolling_success = deque(maxlen=200)
    rolling_rewards = deque(maxlen=200)
    episode = 0
    true_successes = []
    undiscounted_rewards = []
    
    load_gin_configs(config_file, gin_bindings)
    
    experiment_data = []
    
    log_dir = os.path.join(log_dir, str(head_idx), str(env_idx))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir,  
                                "{}.log".format(datetime.datetime.now()))
    logging.basicConfig(filename=log_file, 
                        format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO)
    
    logging.info("Starting policy training for head idx {} seed {}".format(head_idx, env_idx))
    while total_steps < max_steps:
        # if video_generator is not None:
        #     video_generator.episode_start()
        obs, info, rand_idx = env.reset(return_rand_state_idx=True)
        term_state = term_states[rand_idx]
        
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float()
        
        if option.check_termination(head_idx, obs, env):
            print("initiation in termination set. Skip train")
            logging.info("initiation in termination set. Skip train")
            break
        
        option_seed = seed
        if learn_new_policy:
            option_seed = env_idx
            
        def option_term_func(x):
            return epsilon_ball_from_list(x,
                                          term_state)
        
        _, _, _, steps, _, option_rewards, _, _, term_accuracy = option.train_policy(head_idx,
                                                                            env,
                                                                            obs,
                                                                            info,
                                                                            option_seed,
                                                                            max_steps=option_timeout,
                                                                            make_video=True,
                                                                            perfect_term=option_term_func)
        
        undiscounted_rewards.append(option_rewards)
        rolling_rewards.append(np.sum(option_rewards))
        episode += 1
        
        total_steps += steps
        
        true_success = epsilon_ball_from_list(env.get_current_position(),
                                              term_state)
        
        rolling_success.append(true_success)
        true_successes.append(true_success)
        
        experiment_data.append({
            "head_idx": head_idx,
            "option_length": steps,
            "steps": total_steps,
            "reward": option_rewards,
            "total_reward": np.sum(option_rewards),
            "true_success": true_success,
            "final_location": env.get_current_position(),
            "env_idx": env_idx,
            "in_term_accuracy": term_accuracy
        })
        
        if episode%10 == 0:
            logging.info("Head idx: {} Episode: {} Total steps: {} average reward: {} true success: {}".format(head_idx,
                                                                                                episode,
                                                                                                total_steps,
                                                                                                np.mean(rolling_rewards),
                                                                                                np.mean(rolling_success)))
    # if video_generator is not None:
    #     video_generator.episode_end("head{}_env{}_{}".format(head_idx, 
    #                                                                 env_idx,
    #                                                                 run_numbers[(head_idx, env_idx)]))
        # run_numbers[(head_idx, env_idx)] += 1
    
    conn.send(experiment_data)
    conn.close()