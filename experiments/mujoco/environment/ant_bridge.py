import os
import copy
import math

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import matplotlib.pyplot as plt


class AntBridgeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=1, env_type='train', randomize_tasks=True, eval=False):

        self._task = task
        self.env_type = env_type
        self.tasks = self.sample_tasks(n_tasks)
        self.random_steps = 5
        self.eval = eval
        # these will get overiden when we call reset_task from the outside.
        self.max_step = 400
        self.goal_reward = 20
        self.outside_reward = -5
        self.floor_width = 1.5
        self.floor_backAndfront_width = 10
        self.colliding_reward = 0
        self.survive_reward = 0
        self.distanceToStartWeight = 20
        self.distanceToEndWeight = 20 * 2
        self.distanceToGoalWeight = 20 * 3
        self.wind_force_coeffecient = 0.5
        self.init_bridge_x = -9.
        self.current_step = 0
        self._outside = False
        self.ob_shape = {"joint": [29]}
        self.ob_type = self.ob_shape.keys()
        self.bridge_start = 4.5 #5-0.5
        self.bridge_end = 21.5 #21+0.5
        self.start_bridge = False
        self.pass_bridge = False
        self.pass_reward = 10
        self.start_reward = 5
        xml_path = os.path.join(os.getcwd(), "portable/policy/envs/assets/ant-bridge.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)
        self.reset_task(0)

    def step(self, a, skillpolicy=None, id=0):
        # self.render()

        yposbefore = self.get_body_com("agent_torso")[1]

        distanceToBristartBefore = self.distance_to_bridgestart()
        distanceToBriendBefore = self.distance_to_bridgeend()
        distanceToGoalBefore = self.distance_to_goal()
        self.do_simulation(a, self.frame_skip)
        # increment 1 step
        self.current_step += 1
        yposafter = self.get_body_com("agent_torso")[1]
        agent_xpos = self.get_body_com("agent_torso")[0]

        # forward_reward = (yposafter - yposbefore) / self.dt

        # if collide with the wall or went outside, then we shall stop and give agent a big penalty.
        outside_reward = 0
        if yposafter >= 5 and yposafter <= 21:
            if agent_xpos < -self.floor_width or agent_xpos >= self.floor_width:
                self._outside = True
                outside_reward = self.outside_reward
        else:
            if agent_xpos < -self.floor_backAndfront_width or agent_xpos >= self.floor_backAndfront_width:
                self._outside = True
                outside_reward = self.outside_reward

        # check if the agent got tipped over
        tipped_over = self.get_body_com("agent_torso")[2] <= 0.3  # or self.get_body_com("agent_torso")[2]>=1

        distanceToBristartAfter = self.distance_to_bridgestart()
        distanceToBriendAfter = self.distance_to_bridgeend()
        distanceToGoalAfter = self.distance_to_goal()
        #distanceToGoalReward = 0
        #distanceDifference = distanceToGoalBefore - distanceToGoalAfter
        #distanceToGoalReward = distanceDifference * self.distanceToGoalWeight
        agent_xpos, agent_ypos, agent_zpos = self.get_body_com("agent_torso")
        bri_start_reward = 0
        bri_end_reward = 0
        if agent_ypos >= self.bridge_end and not self.pass_bridge:
            bri_end_reward += self.pass_reward
            self.pass_bridge = True

        if distanceToBristartAfter <= 0.5 and not self.start_bridge:
            bri_start_reward += self.start_reward
            self.start_bridge = True
        distanceTostartreward = 0
        if not self.start_bridge:
            distanceDifference = distanceToBristartBefore - distanceToBristartAfter
            distanceTostartreward = distanceDifference * self.distanceToStartWeight

        distanceToendreward = 0
        if self.start_bridge and not self.pass_bridge:
            distanceDifference = distanceToBriendBefore - distanceToBriendAfter
            distanceToendreward = distanceDifference * self.distanceToEndWeight

        distanceToGoalReward = 0
        if self.pass_bridge:
            distanceDifference = distanceToGoalBefore - distanceToGoalAfter
            distanceToGoalReward = distanceDifference * self.distanceToGoalWeight

        # control cost for the agent, I don't think we need it because the ball will just move, leave it for now with a smaller weight.
        ctrl_cost = 0  # 0.5 * np.square(a).sum()
        # I don't think we will have a contact cost ever.

        contact_cost = (
                0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = self.survive_reward

        state = self.state_vector()
        # check when we need to finish the current episode
        done = False
        if self._outside or self.current_step >= self.max_step or tipped_over:
            done = True
        ob = self._get_obs()
        #print("observation:", ob)
        goal_reward = 0
        success = False
        if distanceToGoalAfter < 0.1:
            done = True
            success = True
            goal_reward = self.goal_reward
        reward = outside_reward + goal_reward + distanceToGoalReward + distanceToendreward + distanceTostartreward + bri_start_reward + bri_end_reward - ctrl_cost
        # cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")
        # plt.imshow(cam)
        # plt.pause(0.01)
        return (
            ob,
            reward,
            done,
            dict(
                reward_ctrl=-ctrl_cost,
                success=success,
                obs_img = self._get_img_obs() if self.eval else self._get_obs(),
                position = (self.sim.data.qpos.flat[0], self.sim.data.qpos.flat[1])
            ),
        )

    # the new observation is [agent position, curb1 y axis position, agent velocity]

    def _get_obs(self):
        # x:0 y:1
        return np.concatenate(
            [
                # [self.sim.data.qpos.flat[0] / 10],
                # np.array([10 - self.sim.data.get_body_xpos('agent_torso')[1]]) / 10,
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:],

            ]
        )

    def _get_img_obs(self):

        cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")

        return cam

    def reset_model(self):

        self.current_step = 0
        #self.wind_force()
        self.put_bridge()
        self._outside = False

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        self.set_state(qpos, qvel)
        for _ in range(int(self.random_steps)):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        return self._get_obs()

    def render_camera(self, imshow=False):
        cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")
        if imshow:
            plt.imshow(cam)
            plt.pause(0.01)
        return cam

    def get_position(self):
        return (self.sim.data.qpos.flat[0], self.sim.data.qpos.flat[1])
    
    def place_ant(self, pos=None):
        self.put_bridge()
        self._outside = False

        qpos = self.init_qpos
        # random position if not specified
        if pos is None:
            pos = (
                self.np_random.uniform(-9, 9, size=1),  # x 
                self.np_random.uniform(0, 5, size=1)  # y
            )
        qpos[0] = pos[0]
        qpos[1] = pos[1]

        self.set_state(qpos, self.init_qvel)

        # random steps to ensure proper dynamic 
        for _ in range(5):
            self.step(self.unwrapped.action_space.sample())

        x, y, z = self.get_body_com("agent_torso")
        return x, y

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def collision_detection(self, ref_name=None, body_name=None):
        assert ref_name is not None
        mjcontacts = self.data.contact

        ncon = self.data.ncon
        collision = False
        for i in range(ncon):
            ct = mjcontacts[i]
            g1, g2 = ct.geom1, ct.geom2
            g1 = self.model.geom_names[g1]
            g2 = self.model.geom_names[g2]

            if body_name is not None:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0) and \
                        (g1.find(body_name) >= 0 or g2.find(body_name) >= 0):
                    collision = True
                    break
            else:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0):
                    collision = True
                    break
        return collision

    def wind_force(self):
        torso_index = self.sim.model._body_name2id["agent_torso"]

        self.sim.data.xfrc_applied[torso_index][0] = self.wind_force_coeffecient

    def put_bridge(self):
        #breakpoint()
        self.sim.model.geom_pos[self.model.geom_name2id("midplane")][0] = copy.deepcopy(self.init_bridge_x)

    def distance_to_bridgestart(self):
        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]
        distance = math.sqrt((agent_x - self.init_bridge_x) ** 2 + (agent_y - self.bridge_start) ** 2)
        return distance

    def distance_to_bridgeend(self):
        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]
        distance = math.sqrt((agent_x - self.init_bridge_x) ** 2 + (agent_y - self.bridge_end) ** 2)
        return distance

    def distance_to_goal(self):
        #goal_x = 0  # self.sim.data.get_geom_xpos('goal')[0]
        goal_y = 26  # self.sim.data.get_geom_xpos('goal')[1]
        #agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        #distance = math.sqrt((goal_x - agent_x) ** 2 + (goal_y - agent_y) ** 2)
        distance = goal_y - agent_y
        return distance

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def sample_tasks(self, num_tasks):

        if self.env_type == 'test':
            wind_force_coeffecients = np.linspace(-2.50, 2.5, num_tasks)
            all_bridge_pos = np.linspace(-8, 8, num_tasks)
            tasks = [{'bridge_pos': bridge_pos} for bridge_pos in
                     all_bridge_pos]
        else:
            wind_force_coeffecients = np.random.uniform(-3.0, 3.0, size=(num_tasks,))
            all_bridge_pos = np.random.uniform(-8.0, 8.0, size=(num_tasks,))
            tasks = [{'bridge_pos': bridge_pos} for bridge_pos in
                     all_bridge_pos]
        return tasks

    def reset_task(self, idx):

        self._task = self.tasks[idx]
        #self.wind_force_coeffecient = self._task['wind_force_coeffecient']
        self.init_bridge_x = self._task['bridge_pos']
        self.reset()
