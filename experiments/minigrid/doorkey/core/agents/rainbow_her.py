import os
import random
import argparse
import numpy as np

from experiments.minigrid.doorkey.core.agents.rainbow import Rainbow
from experiments.minigrid.utils import environment_builder, determine_goal_pos


def create_agent(
        n_actions,
        gpu,
        n_input_channels,
        env_steps=500_000,
        lr=1e-4,
        sigma=0.5
    ):
    kwargs = dict(
        n_atoms=51, v_max=10., v_min=-10.,
        noisy_net_sigma=sigma, lr=lr, n_steps=3,
        betasteps=env_steps // 4,
        replay_start_size=1024, 
        replay_buffer_size=int(3e5),
        gpu=gpu, n_obs_channels=2*n_input_channels,
        use_custom_batch_states=False,
        epsilon_decay_steps=args.epsilon_decay_steps
    )
    return Rainbow(n_actions, **kwargs)


def concat(obs, goal):
    return np.concatenate((obs, goal), axis=0)


def rollout(
        agent: Rainbow, env, obs: np.ndarray, goal: np.ndarray, goal_info: dict
    ):
    global goal_observation

    done = False
    episode_reward = 0.
    rewards = []
    trajectory = []
    
    while not done:
        action = agent.act(concat(obs, goal))
        next_obs, reward, done, info = env.step(action)
        rewards.append(reward)
        trajectory.append((obs, action, reward, next_obs, info))

        if reward == 1:
            goal_observation = next_obs

        obs = next_obs
        episode_reward += reward
    
    # HER
    experience_replay(agent, trajectory, goal, goal_info)
    hindsight_goal, hindsight_goal_info = pick_hindsight_goal(trajectory)
    experience_replay(agent, trajectory, hindsight_goal, hindsight_goal_info)

    return obs, info, rewards


def experience_replay(agent: Rainbow, transitions, goal, goal_info):
    relabeled_trajectory = []
    for state, action, _, next_state, info in transitions:
        sg = concat(state, goal)
        nsg = concat(next_state, goal)
        reached = info['player_pos'] == goal_info['player_pos']
        reward = float(reached) + (args.exploration_bonus_scale * info['bonus'])
        relabeled_trajectory.append((
        sg, action, reward, nsg, reached, info['needs_reset']))
        if reached:
            break
    agent.experience_replay(relabeled_trajectory)


def select_goal(task_goal, task_goal_info, method='task'):
    if method == 'task':
        return task_goal, task_goal_info
    raise NotImplementedError


def pick_hindsight_goal(transitions, method='final'):
    if method == 'final':
        goal_transition = transitions[-1]
        goal = goal_transition[-2]
        goal_info = goal_transition[-1]
        return goal, goal_info
    if method == 'future':
        start_idx = len(transitions) // 2
        goal_idx = random.randint(start_idx, len(transitions) - 1)
        goal_transition = transitions[goal_idx]
        goal = goal_transition[-2]
        goal_info = goal_transition[-1]
        return goal, goal_info
    raise NotImplementedError(method)


def train2(agent: Rainbow, env,
           task_goal: np.ndarray, task_goal_info: dict,
           start_episode, n_episodes):
    rewards = []
    for episode in range(start_episode, start_episode + n_episodes):
        obs0, info0 = env.reset()
        assert not info0['needs_reset'], info0
        goal, goal_info = select_goal(task_goal, task_goal_info)
        state, info, episode_rewards = rollout(agent, env, obs0, goal, goal_info)
        undiscounted_return = sum(episode_rewards)
        rewards.append(undiscounted_return)
        print(100 * '-')
        print(f'Episode: {episode}',
        f"InitPos': {info0['player_pos']}",
        f"GoalPos: {goal_info_dict['player_pos']}",
        f"FinalPos: {info['player_pos']}",
        f'Reward: {undiscounted_return}')
        print(100 * '-')
    return rewards


def test(agent: Rainbow, env,
         task_goal: np.ndarray, task_goal_info: dict,
         n_episodes):
    rewards = []
    for episode in range(n_episodes):
        env.reset()
        obs0, info0 = env.reset_to(start_info['player_pos'])
        assert not info0['needs_reset'], info0
        goal, goal_info = select_goal(task_goal, task_goal_info)
        state, info, episode_rewards = rollout(agent, env, obs0, goal, goal_info)
        undiscounted_return = sum(episode_rewards)
        rewards.append(undiscounted_return)
        print(100 * '-')
        print(f'[Test] Episode: {episode}',
        f"InitPos': {info0['player_pos']}",
        f"GoalPos: {goal_info_dict['player_pos']}",
        f"FinalPos: {info['player_pos']}",
        f'Reward: {undiscounted_return}')
        print(100 * '-')
    return np.mean(rewards)


# def log(episode, msr):
#     global success_rates
#     success_rates.append(msr)
#     fname = f'{g_log_dir}/log_seed{args.seed}.pkl'
#     utils.safe_zip_write(fname, {'current_episode': episode,
#                                 'rewards': success_rates})


def run_with_eval(agent: Rainbow, env, n_iterations,
        task_goal: np.ndarray, task_goal_info: dict):
    current_episode = 0
    for iter in range(n_iterations):
        train2(agent, env, task_goal, task_goal_info, current_episode, 10)
        msr = test(agent, env, task_goal, task_goal_info, 5)
        print(f'[EvaluationIter={iter}] Mean Success Rate: {msr}')
        current_episode += 10
        # log(current_episode, msr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--sub_dir', type=str, default='', help='sub dir for sweeps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--environment_name', type=str, default='MiniGrid-Empty-8x8-v0')
    parser.add_argument('--n_iterations', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--exploration_bonus_scale', type=float, default=0)
    parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
    parser.add_argument('--epsilon_decay_steps', type=int, default=12_500)
    parser.add_argument('--use_random_resets', action="store_true", default=False)
    parser.add_argument('--n_actions', type=int, help='Specify the number of actions in the env')
    args = parser.parse_args()

    g_log_dir = os.path.join(args.log_dir, args.experiment_name, args.sub_dir)

    # utils.create_log_dir(args.log_dir)
    # utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name))
    # utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name, args.sub_dir))
    # utils.create_log_dir(g_log_dir)

    # utils.set_random_seed(args.seed)

    environment = environment_builder(
        level_name=args.environment_name,
        seed=args.seed,
        exploration_reward_scale=0,
        random_reset=args.use_random_resets
    )

    rainbow_agent = create_agent(
        environment.action_space.n if args.n_actions is None else args.n_actions,
        gpu=args.gpu,
        n_input_channels=1,
        lr=args.lr,
        sigma=args.sigma
    )

    s0, info0 = environment.reset()

    start_state = environment.official_start_obs
    start_info = environment.official_start_info

    goal_info_dict = dict(player_pos=determine_goal_pos(environment))
    goal_observation = np.zeros_like(s0)

    success_rates = []

    run_with_eval(
        rainbow_agent,
        environment,
        args.n_iterations,
        goal_observation,
        goal_info_dict
    )