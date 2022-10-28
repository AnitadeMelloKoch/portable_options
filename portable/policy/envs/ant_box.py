import os
import math
import random

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from matplotlib import pyplot as plt


class AntBoxEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=1, env_type='train', randomize_tasks=True, eval=False):

        self._task = task
        self.env_type = env_type
        self.tasks = self.sample_tasks(n_tasks)
        self.random_steps = 5
        self.eval = eval
        # these will get overiden when we call reset_task from the outside.
        self.max_step = 400
        self.outside_reward = 0
        self.floor_width = 10
        self.floor_backAndfront_width = 10
        self._passing_cliff = False
        self._next_to_box = False
        self.colliding_reward = 0
        self.survive_reward = 0
        self.distanceToGoalWeight = 20 * 4
        self.distanceToBoxWeight = 20
        self.distanceToCliffWeight = 20 * 6
        self.speedPenaltyWeight = 3
        self._init_box_x_position = 0
        self._init_box_y_position = 8
        self._box_z_position = 0.5
        self.pass_reward = 20
        self.goal_reward = 40
        self.cliff_end_ypos = 14.5


        self.current_step = 0
        self._outside = False
        self.ob_shape = {"joint": [29]}
        self.ob_type = self.ob_shape.keys()
        xml_path = os.path.join(os.getcwd(), "portable/policy/envs/assets/ant-box.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a, skillpolicy=None, id=0):

        distanceToBoxBefore = self.box_distance_to_ant()
        distanceToGoalBefore = self.distance_to_goal()
        distanceToCliffEndBefore = self.distance_to_cliff_end()
        xposbefore , yposbefore,  zposbefore= self.get_body_com("agent_torso")

        self.do_simulation(a, self.frame_skip)
        # increment 1 step
        self.current_step += 1
        agent_xpos, agent_ypos, agent_zpos = self.get_body_com("agent_torso")
        forward_speed_x = (agent_xpos - xposbefore) / self.dt
        forward_speed_y = (agent_ypos - yposbefore) / self.dt
        speed = np.sqrt(forward_speed_x*forward_speed_x + forward_speed_y*forward_speed_y)
        #print("current speed:", speed)
        distanceToBoxAfter = self.box_distance_to_ant()
        distanceToGoalAfter = self.distance_to_goal()
        distanceToCliffEndAfter = self.distance_to_cliff_end()
        #if the agent has crossed the middle of the cliff, we labeled it
        passing_cliff_reward = 0
        if agent_ypos >=self.cliff_end_ypos and not self._passing_cliff:
            passing_cliff_reward += self.pass_reward
            self._passing_cliff = True

        if distanceToBoxAfter <= 1.2:
            self._next_to_box = True

        distanceToBoxReward = 0

        if not self._next_to_box:
            distanceDifference = distanceToBoxBefore - distanceToBoxAfter
            speed_penalty =  0
            distanceToBoxReward = distanceDifference * self.distanceToBoxWeight - speed_penalty * self.speedPenaltyWeight

        distanceToCliffReward = 0
        if self._next_to_box and not self._passing_cliff:
            distanceDifference = distanceToCliffEndBefore - distanceToCliffEndAfter
            speed_penalty = max(0, speed-1)
            distanceToCliffReward = distanceDifference * self.distanceToCliffWeight - speed_penalty * self.speedPenaltyWeight * 2

        distanceToGoalReward = 0
        # if passed the cliff, will give it some dense reward to motivate the agent to get to the goal
        if self._passing_cliff:
            distanceDifference = distanceToGoalBefore - distanceToGoalAfter
            distanceToGoalReward = distanceDifference * self.distanceToGoalWeight

        outside_reward = 0
        #went outside or fall off the cliff
        if abs(agent_xpos) >=self.floor_width or agent_zpos <0:
            self._outside = True
            outside_reward = self.outside_reward


        # check if the agent got tipped over
        tipped_over = self.get_body_com("agent_torso")[2] <= 0.3  # or self.get_body_com("agent_torso")[2]>=1

        # control cost for the agent, I don't think we need it because the ball will just move, leave it for now with a smaller weight.
        ctrl_cost = - 0.005 * np.square(a).sum()
        # I don't think we will have a contact cost ever.

        contact_cost = (
                0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = self.survive_reward

        # check when we need to finish the current episode
        done = False
        if self._outside or self.current_step >= self.max_step or tipped_over:
            done = True


        ob = self._get_obs()
        # print(ob.shape)
        goal_reward = 0
        success = False
        # cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")
        # plt.imshow(cam)
        # plt.pause(0.01)
        if distanceToGoalAfter < 0.1:
            done = True
            success = True
            goal_reward = self.goal_reward
        reward = outside_reward + goal_reward + distanceToGoalReward + distanceToBoxReward + ctrl_cost + distanceToCliffReward + passing_cliff_reward
        #print("speed:", speed, "ctrl_cost:", ctrl_cost, "distanceToBoxReward", distanceToBoxReward, "distanceToCliffReward", distanceToCliffReward, "distanceToGoalReward", distanceToGoalReward, "passing_cliff_reward", passing_cliff_reward )
        return (
            ob,
            reward,
            done,
            dict(
                reward_ctrl=-ctrl_cost,
                success=success,
                obs_img = self._get_img_obs() if self.eval else self._get_obs(),
            ),
        )

    # the new observation is [agent position, curb1 y axis position, agent velocity]
    def _get_img_obs(self):

        cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")

        return cam

    def _get_obs(self):

        obs =  np.concatenate(
            [
                #box y position agent_torso_geom
                #box z position
                # [self.sim.data.qpos.flat[self.sim.model.get_joint_qpos_addr('OBJTz')]],
                #agent x position
                # [np.array(self.sim.data.get_body_xpos('agent_torso')[0]) / 6],
                #agent y position relative to the begining of the cliff
                # [np.array(8 - self.sim.data.get_body_xpos('agent_torso')[1]) / 8],
                # [(self.sim.data.qpos.flat[self.sim.model.get_joint_qpos_addr('OBJTy')] - 10) / 2],
                # [self._next_to_box],
                # [self._passing_cliff],
                #agent z position
                self.sim.data.qpos.flat[4:],
                #speed
                self.sim.data.qvel.flat[3:],

            ]
        )
        return obs

    def reset_model(self):
        self.push_box()
        id = random.randint(0,len(self.tasks)-1)
        self._task = self.tasks[id]
        self.current_step = 0
        self._outside = False
        self._passing_cliff = False
        self._next_to_box = False

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        qpos, qvel = self._put_box(qpos, qvel)

        self.set_state(qpos, qvel)

        for _ in range(int(self.random_steps)):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")
        ob = self._get_obs()
        return ob

    def push_box(self):
        box_index = self.sim.model._body_name2id["box"]
        self.sim.data.xfrc_applied[box_index][2] = -100

    def render_camera(self, imshow=False):
        cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")
        if imshow:
            plt.imshow(cam)
            plt.pause(0.01)
        return cam
    
    def place_ant(self, pos=None):
        self.push_box()
        qpos = self.init_qpos
        # random position if not specified
        if pos is None:
            pos = (
                self.np_random.uniform(-9, 9, size=1),  # x 
                self.np_random.uniform(-10, 25, size=1)  # y
            )

        qpos[2] = pos[0]
        qpos[3] = pos[1]
        qvel = self.init_qvel

        qpos, qvel = self._put_box(qpos, qvel)
        self.set_state(qpos, qvel)

        # random steps to ensure proper dynamic 
        for _ in range(8):
            self.step(self.unwrapped.action_space.sample())
        
        x, y, z = self.get_body_com("agent_torso")
        return x, y

    def _put_box(self, qpos, qvel):
        pos_y_joint, pos_z_joint  = self.get_box_joint_pos()
        #pos_x_joint, pos_y_joint, pos_z_joint  = self.get_box_joint_pos()
        #vel_x_joint, vel_y_joint, vel_z_joint = self.get_box_joint_vel()
        vel_y_joint, vel_z_joint = self.get_box_joint_vel()
        #qpos[pos_x_joint] = self._init_box_x_position
        self.model.body_pos[self.model.body_name2id("box")][0] = self._init_box_x_position
        qpos[pos_y_joint] = self._init_box_y_position
        qpos[pos_z_joint] = self._box_z_position
        #qvel[vel_x_joint] = 0
        qvel[vel_y_joint] = 0
        qvel[vel_z_joint] = 0
        return qpos, qvel

    def box_distance_to_ant(self):
        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]
        #box_x = self.get_body_com("box")[0]
        #box_y = self.get_body_com("box")[1] - 2
        distance =  math.sqrt((agent_x - self._init_box_x_position) **2 + (agent_y - self._init_box_y_position + 4) **2)
        return distance

    def distance_to_cliff_end(self):
        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]
        cliff_center_x = self._init_box_x_position
        cliff_center_y = self.cliff_end_ypos
        distance =  math.sqrt((agent_x - cliff_center_x) **2 + (agent_y - cliff_center_y) **2)
        return distance


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


    def distance_to_goal(self):
        #goal_x = 0  # self.sim.data.get_geom_xpos('goal')[0]
        goal_y = 16  # self.sim.data.get_geom_xpos('goal')[1]
        #agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        #distance = math.sqrt((goal_x - agent_x) ** 2 + (goal_y - agent_y) ** 2)
        distance = goal_y-agent_y
        return distance

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def get_box_joint_pos(self):

        # joint_OBJTx = self.sim.model.get_joint_qpos_addr('OBJTx')
        joint_OBJTy = self.sim.model.get_joint_qpos_addr('OBJTy')
        joint_OBJTz = self.sim.model.get_joint_qpos_addr('OBJTz')
        return joint_OBJTy, joint_OBJTz
        #return joint_OBJTx, joint_OBJTy, joint_OBJTz

    def get_box_joint_vel(self):
        # joint_OBJTx = self.sim.model.get_joint_qvel_addr('OBJTx')
        joint_OBJTy = self.sim.model.get_joint_qvel_addr('OBJTy')
        joint_OBJTz = self.sim.model.get_joint_qvel_addr('OBJTz')
        return joint_OBJTy, joint_OBJTz
        #return joint_OBJTx, joint_OBJTy, joint_OBJTz


    def sample_tasks(self, num_tasks):

        if self.env_type == 'test':
            box_x_positions = np.linspace(-6, 6, num_tasks)
        else:
            box_x_positions = np.random.uniform(-6, 6, size=(num_tasks,))

        box_y_positions = np.random.uniform(7.999, 8.001, size=(num_tasks,))
        tasks = [{'box_x_position': box_x_position, 'box_y_position': box_y_position} for box_x_position, box_y_position in zip(box_x_positions, box_y_positions)]

        return tasks



    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._init_box_x_position = self._task['box_x_position']
        self._init_box_y_position = self._task['box_y_position']

        self.reset()
