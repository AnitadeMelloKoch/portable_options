import os
import copy
import math

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import matplotlib.pyplot as plt

ANT_GATHER_LENGTH = 16
ANT_GOAL_LENGTH = 25
ANT_BRIDGE_LENGTH = 26
ANT_BOX_LENGTH = 26


class AntMixLongEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=1, env_type='train', randomize_tasks=True, eval=False):
        self.eval = eval
        self._task_sets = ["antgoal0", "antbrid0", "antbox0", "antgoal1", "antbrid1", "antbox1"]
        #self.task_order = np.arange(15)
        self.task_order = np.random.choice(6, 6, replace=False)
        self._task = task
        self.subtaskid = 0
        #gather 5 ant bridge 3 ant goal 4 antbox 3
        self.subtasktypes = ["antgoal", "antbridge", "antbox", "antgoal", "antbridge", "antbox"]
        self.env_type = env_type
        # self.tasks = self.sample_tasks(n_tasks)
        # these will get overiden when we call reset_task from the outside.
        self._x_pos_sampler = 0.5
        self._curb_y_pos = 10
        self.random_steps = 1
        self.max_step = 5000
        self.passing_reward = 10
        self.passing_door = [0, 0]
        self._next_to_box = [0, 0]
        self._passing_cliff = [0, 0]
        self.passing_reward = 10
        self.goal_reward = 20
        self.outside_reward = -5
        self.floor_width = 10
        if env_type == 'train':
            self.corridor_width = 3
        elif env_type == 'test':
            self.corridor_width = 3
        self.corridor_pos = [0, 0, 0, 0]
        self.survive_reward = 0
        self.count_down = 0
        self.current_step = 0
        self.windforce = [0, 0, 0]
        self.coin_pos = [0, 0, 0, 0, 0]
        self.box_pos = [0, 0, 0]
        self._box_z_position = 0.5
        self._init_box_y_position = 8
        self.coin_reward_weight = 0
        self.substask_succeed_weight = 5
        self.success_reward_weight = 100
        self.speedPenaltyWeight = 3
        self.goals_position_y = [25, 51, 67, 93, 118, 144, 160, 186, 211, 237, 253, 279, 304, 320, 336]
        self.offset_y = [0 + 10, 25 + 10, 51 + 8, 67 + 10, 93 + 10,
                         118 + 10, 144 + 8, 160 + 10, 186 + 10, 211 + 10, 237 + 8,
                         253 + 10, 279 + 10, 304 + 8, 320 + 8]

        self.first_coins_get = [0, 0, 0, 0, 0]
        self.second_coins_get = [0, 0, 0, 0, 0]
        self.ob_shape = {"joint": [29]}
        self.ob_type = self.ob_shape.keys()
        xml_path = os.path.join(os.getcwd(), "portable/procgen/envs/assets/ant-mix-option.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a, skillpolicy=False, id=None):
        # do the simulation and check how much we moved on the y axis
        xposbefore, yposbefore, zposbefore = self.get_body_com("agent_torso")
        id_subtask = int(self._task_sets[self.task_order[self.subtaskid]][-1])
        if self.subtasktypes[self.subtaskid] == "antbridge":
            id_bridgetask = int(self._task_sets[self.task_order[self.subtaskid]][-1])
            force = self.windforce[id_bridgetask]
            self.wind_force(force)
        else:
            self.wind_force(0)
        distanceToGoalBefore = self.distance_to_goal(self.goals_position_y[self.subtaskid])
        #distanceToDoorBefore will only be correct when the current task is a door task
        distanceToDoorBefore = self.distance_to_door(id_subtask, self.offset_y[self.subtaskid])
        # first_coin_distance_Before, second_coin_distance_Before, coin_task_no = self.distance_to_coins()
        distanceToBoxBefore = self.box_distance_to_ant(id_subtask, self.offset_y[self.subtaskid])
        distanceToCliffBefore = self.distance_to_cliff_end(id_subtask, self.offset_y[self.subtaskid])
        self.do_simulation(a, self.frame_skip)

        agent_xpos, agent_ypos, agent_zpos = self.get_body_com("agent_torso")
        distanceToGoalAfter = self.distance_to_goal(self.goals_position_y[self.subtaskid])
        forward_speed_x = (agent_xpos - xposbefore) / self.dt
        forward_speed_y = (agent_ypos - yposbefore) / self.dt
        speed = np.sqrt(forward_speed_x*forward_speed_x + forward_speed_y*forward_speed_y)

        tipped_over = self.get_body_com("agent_torso")[2] <= 0.3
        subtask_succeed = False
        get_coin_reward = 0

        if self.subtasktypes[self.subtaskid] == "antgoal":
            # if we haven't passed the door
            if self.passing_door[id_subtask] ==0:
                distanceToDoorAfter = self.distance_to_door(id_subtask, self.offset_y[self.subtaskid])
                distanceDifference = distanceToDoorBefore - distanceToDoorAfter
                dense_reward = distanceDifference * 10
                # if the agent hasn't pass the door and agent y postion is bigger than the door, passed
                if self.sim.data.get_body_xpos('agent_torso')[1] >= self.offset_y[self.subtaskid]:
                    self.passing_door[id_subtask] = 1
            else:
                distanceDifference = distanceToGoalBefore - distanceToGoalAfter
                dense_reward = distanceDifference * 10
            if distanceToGoalAfter <= 2:
                subtask_succeed = True

        elif self.subtasktypes[self.subtaskid] == "antbridge":
            dense_reward = (distanceToGoalBefore - distanceToGoalAfter) * 10
            if distanceToGoalAfter <= 1.2:
                subtask_succeed = True
        elif self.subtasktypes[self.subtaskid] == "antbox":
            distanceToBoxAfter = self.box_distance_to_ant(id_subtask, self.offset_y[self.subtaskid])
            distanceToCliffAfter = self.distance_to_cliff_end(id_subtask, self.offset_y[self.subtaskid])
            # if we haven't reached to the box
            if self._next_to_box[id_subtask] ==0:
                distanceDifference = distanceToBoxBefore - distanceToBoxAfter
                speed_penalty = max(0, speed - 2)
                distanceToBoxReward = distanceDifference * 10- speed_penalty * self.speedPenaltyWeight
                dense_reward = distanceToBoxReward
                if distanceToBoxAfter <= 0.8:
                    self._next_to_box[id_subtask] = 1

            elif self._next_to_box[id_subtask] ==1 and self._passing_cliff[id_subtask] == 0:
                distanceDifference = distanceToCliffBefore - distanceToCliffAfter
                speed_penalty = max(0, speed - 1)
                distanceToCliffReward = distanceDifference * 10 - speed_penalty * self.speedPenaltyWeight * 2
                dense_reward = distanceToCliffReward
                if agent_ypos >= self.offset_y[self.subtaskid] + 4.5:
                    self._passing_cliff[id_subtask] = 1
            else:
                distanceDifference = distanceToGoalBefore - distanceToGoalAfter
                dense_reward = distanceDifference * 10

            if distanceToGoalAfter <= 2:
                subtask_succeed = True
        # else:
        #     first_coin_distance, second_coin_distance, coin_task_no = self.distance_to_coins()
        #     if self.first_coins_get[coin_task_no] == 0:
        #         dense_reward = (first_coin_distance_Before - first_coin_distance) * 10
        #         if first_coin_distance <= 1.2:
        #             get_coin_reward += self.coin_reward_weight
        #             self.first_coins_get[coin_task_no] = 1
        #             self.put_firstCoin_away(coin_task_no)
        #
        #     if self.second_coins_get[coin_task_no] == 0 and self.first_coins_get[coin_task_no] == 1:
        #         dense_reward = (second_coin_distance_Before - second_coin_distance) * 10
        #         if second_coin_distance <= 1.2:
        #             get_coin_reward += self.coin_reward_weight
        #             self.second_coins_get[coin_task_no] = 1
        #             self.put_secondCoin_away(coin_task_no)
        #
        #     if self.first_coins_get[coin_task_no] and self.second_coins_get[coin_task_no]:
        #         dense_reward = (distanceToGoalBefore - distanceToGoalAfter) * 10
        #         if distanceToGoalAfter <= 1.3:
        #             subtask_succeed = True

        state = self.state_vector()
        # check when we need to finish the current episode
        done = False
        self.count_down += 1
        if self.count_down >500:
            done = True
        self.current_step += 1
        if tipped_over or self.current_step >= self.max_step:
            done = True
        goal_reward = 0
        success = False
        substask_reward = 0
        if not skillpolicy:
            ob = self._get_obs()
        else:
            ob = self._get_obs_sub(id)
        if subtask_succeed:
            self.count_down = 0
            self.subtaskid += 1
            substask_reward += self.substask_succeed_weight
        success_reward = 0
        if self.subtaskid == 6:
            success = True
            done = True
            success_reward += self.success_reward_weight
            #self.subtaskid = 14

        reward = success_reward + substask_reward*2 + get_coin_reward + dense_reward
        # cam = self.render(mode="rgb_array", width=1280, height=1280)
        # plt.imshow(cam)
        # plt.pause(1)
        return (
            ob,
            reward,
            done,
            dict(
                success=success,
                subtask_success=subtask_succeed,
                sparse_reward=success_reward + substask_reward,
                obs_img = self._get_img_obs() if self.eval else self._get_obs(),
            ),
        )

    # the new observation is [agent position, curb1 y axis position, agent velocity]

    def _get_img_obs(self):
        cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")
        return cam

    def _get_obs_sub(self, id=None):
        relative_y = 0
        if id == 2:
            relative_y = np.array([self.offset_y[self.subtaskid] - self.sim.data.get_body_xpos('agent_torso')[1]]) / 8
            _, _, coin_task_no = self.distance_to_coins()
            return np.concatenate(
                [
                    [self.sim.data.qpos.flat[6] / 6],
                    relative_y,
                    [self.first_coins_get[coin_task_no]],
                    [self.second_coins_get[coin_task_no]],
                    self.sim.data.qpos.flat[8:],
                    self.sim.data.qvel.flat[6:] / 5,

                ]
            )
        elif id == 0 or id == 1:
            relative_y = np.array([self.offset_y[self.subtaskid] - self.sim.data.get_body_xpos('agent_torso')[1]]) / 10
            return np.concatenate(
                [
                    [self.sim.data.qpos.flat[6] / 10],
                    relative_y,
                    self.sim.data.qpos.flat[8:],
                    self.sim.data.qvel.flat[6:] / 5,

                ]
            )
        elif id == 3:
            ith_antbox= int(self._task_sets[self.task_order[self.subtaskid]][-1])
            if ith_antbox >=3:
                ith_antbox =2
            relative_y = np.array([self.offset_y[self.subtaskid] - self.sim.data.get_body_xpos('agent_torso')[1]]) / 10
            return np.concatenate(
                [
                    [(self.sim.data.qpos.flat[self.sim.model.get_joint_qpos_addr('OBJTy'+str(ith_antbox))]-self.offset_y[self.subtaskid])/2],
                    [np.array(self.sim.data.get_body_xpos('agent_torso')[0]) / 10],
                    relative_y,
                    self.sim.data.qpos.flat[9:],
                    self.sim.data.qvel.flat[7:] / 5,

                ]
            )

    def _get_obs(self):
        task_type_onehot = np.zeros(4)
        x_pos = 0
        # ith_subtask = int(self._task_sets[self.task_order[self.subtaskid]][-1])
        # if self.subtasktypes[self.subtaskid] == "antgoal":
        #     task_type_onehot[0] = 1
        #     x_pos = self.corridor_pos[ith_subtask] / 3  # normalization
        #
        # elif self.subtasktypes[self.subtaskid] == "antbridge":
        #     task_type_onehot[1] = 1
        #     x_pos = self.windforce[ith_subtask] / 2  # normalization
        # elif self.subtasktypes[self.subtaskid] == "antgather":
        #     x_pos = self.coin_pos[ith_subtask] / 3  # normalization
        #     task_type_onehot[2] = 1
        # elif self.subtasktypes[self.subtaskid] == "antbox":
        #     x_pos = self.coin_pos[ith_subtask] / 3  # normalization
        #     task_type_onehot[3] = 1
        task_id_onehot = np.zeros(6)
        if self.subtaskid != 6:
            task_id_onehot[self.subtaskid] = 1
        return np.concatenate(
            [
                # [self.sim.data.qpos.flat[6] / 10],
                # np.array([self.sim.data.get_body_xpos('agent_torso')[1] - 168]) / 16.8,
                #
                # task_id_onehot,
                self.sim.data.qpos.flat[6:],
                self.sim.data.qvel.flat[4:],
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        qpos, qvel = self.combine_subtask(qpos, qvel, fixed=True)
        self.push_box()
        self.current_step = 0
        self.first_coins_get = [0, 0, 0, 0, 0]
        self.second_coins_get = [0, 0, 0, 0, 0]
        self.windforce = [-2,0,2]
        #self.windforce = [0,0,0]
        #self.windforce = np.random.uniform(-2, 2, 3)
        self.corridor_pos = [0, 0, 0, 0]
        self.coin_pos = [0, 0, 0, 0, 0]
        self.box_pos = [0,0,0]
        self.subtaskid = 0

        self.set_state(qpos, qvel)

        # for _ in range(int(self.random_steps)):
        #     self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        return self._get_obs()

    def combine_subtask(self, qpos, qvel, fixed=True):
        if not fixed:
            self.task_order = np.random.choice(6, 6, replace=False)

        self.goals_position_y = []
        self.offset_y = []
        self.subtasktypes = []
        last_goal_position = 0
        all_bridge_pos = np.random.uniform(-8.0, 8.0, size=(2,))
        for task_no in self.task_order:
            if self._task_sets[task_no][:6] == "antgoa":
                #print("1")
                idx_plane = self.model.geom_name2id("antgoal" + str(self._task_sets[task_no][-1]) + "_plane")
                idx_curbleft = self.model.geom_name2id("curbleft" + str(self._task_sets[task_no][-1]))
                idx_curbright = self.model.geom_name2id("curbright" + str(self._task_sets[task_no][-1]))
                ith_antgoal = int(self._task_sets[task_no][-1])
                self._set_curbs_xposition(idx_curbleft, idx_curbright, ith_antgoal)
                self.sim.model.geom_pos[idx_plane][1] = last_goal_position + ANT_GOAL_LENGTH / 2
                self.sim.model.geom_pos[idx_curbleft][1] = last_goal_position + 10
                self.sim.model.geom_pos[idx_curbright][1] = last_goal_position + 10
                self.goals_position_y.append(ANT_GOAL_LENGTH + last_goal_position)
                self.offset_y.append(last_goal_position + 10)
                self.subtasktypes.append("antgoal")

            elif self._task_sets[task_no][:6] == "antbri":
                #print("2")
                idx_frontplane = self.model.geom_name2id(
                    "antbridge" + str(self._task_sets[task_no][-1]) + "_frontplane")
                idx_rearplane = self.model.geom_name2id("antbridge" + str(self._task_sets[task_no][-1]) + "_rearplane")
                idx_bridge = self.model.geom_name2id("bridge" + str(self._task_sets[task_no][-1]))

                self.sim.model.geom_pos[idx_frontplane][1] = last_goal_position + 2.5
                self.sim.model.geom_pos[idx_bridge][0] = copy.deepcopy(all_bridge_pos[int(self._task_sets[task_no][-1])])
                self.sim.model.geom_pos[idx_bridge][1] = last_goal_position + 5 + 8
                self.sim.model.geom_pos[idx_rearplane][1] = last_goal_position + 5 + 16 + 2.5
                self.goals_position_y.append(ANT_BRIDGE_LENGTH + last_goal_position)
                self.offset_y.append(last_goal_position + 10)
                self.subtasktypes.append("antbridge")

            elif self._task_sets[task_no][:6] == "antgat":

                idx_plane = self.model.geom_name2id("antgather" + str(self._task_sets[task_no][-1]) + "_plane")
                self.sim.model.geom_pos[idx_plane][1] = last_goal_position + ANT_GATHER_LENGTH / 2
                coin1 = "coin_geom1_" + str(self._task_sets[task_no][-1])
                coin2 = "coin_geom2_" + str(self._task_sets[task_no][-1])
                ith_antgather = int(self._task_sets[task_no][-1])

                self.set_coins(last_goal_position, coin1, coin2, ith_antgather)
                self.goals_position_y.append(ANT_GATHER_LENGTH + last_goal_position)
                self.offset_y.append(last_goal_position + 8)
                self.subtasktypes.append("antgather")
            elif self._task_sets[task_no][:6] == "antbox":
                #print("3")
                idx_frontplane = self.model.geom_name2id(
                    "antbox" + str(self._task_sets[task_no][-1]) + "_frontplane")
                idx_backplane = self.model.geom_name2id(
                    "antbox" + str(self._task_sets[task_no][-1]) + "_backplane")
                idx_midplane = self.model.geom_name2id(
                    "antbox" + str(self._task_sets[task_no][-1]) + "_midplane")
                ith_antbox = int(self._task_sets[task_no][-1])

                qpos, qvel = self.put_box(last_goal_position, ith_antbox, qpos, qvel)
                self.sim.model.geom_pos[idx_frontplane][1] = last_goal_position + 5
                self.sim.model.geom_pos[idx_backplane][1] = last_goal_position + 20
                self.sim.model.geom_pos[idx_midplane][1]  = last_goal_position + 12
                self.goals_position_y.append(ANT_BOX_LENGTH + last_goal_position)
                self.offset_y.append(last_goal_position + 10)
                self.subtasktypes.append("antbox")
            else:
                raise NameError('Wrong subtask type')

            last_goal_position = self.goals_position_y[-1]

        return qpos, qvel

    def set_coins(self, start_position, coin1, coin2, ith_antgather):
        idx1 = self.model.geom_name2id(coin1)
        idx2 = self.model.geom_name2id(coin2)
        # first_coin_x_position = 3
        if ith_antgather == 0:
            first_coin_x_position = 3
        elif ith_antgather == 1:
            first_coin_x_position =  2
        elif ith_antgather == 2:
            first_coin_x_position =  1
        elif ith_antgather == 3:
            first_coin_x_position =  -2
        elif ith_antgather == 4:
            first_coin_x_position =  -3
        else:
            first_coin_x_position = -100
            raise NameError('Wrong subtask type')
        #first_coin_x_position = np.random.uniform(-3, 3)
        first_coin_position = (first_coin_x_position, 5)
        second_coin_position = (-first_coin_x_position, 11)
        self.coin_pos[ith_antgather] = first_coin_x_position
        self.sim.model.geom_pos[idx1][0] = first_coin_position[0]
        self.sim.model.geom_pos[idx1][1] = first_coin_position[1] + start_position
        self.sim.model.geom_pos[idx2][0] = second_coin_position[0]
        self.sim.model.geom_pos[idx2][1] = second_coin_position[1] + start_position

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    #add a force on the top of each box
    def push_box(self):
        box0_index = self.sim.model._body_name2id["box0"]
        box1_index = self.sim.model._body_name2id["box1"]
        # box2_index = self.sim.model._body_name2id["box2"]
        self.sim.data.xfrc_applied[box0_index][2] = -100
        self.sim.data.xfrc_applied[box1_index][2] = -100
        # self.sim.data.xfrc_applied[box2_index][2] = -100

    def put_box(self, last_goal_position, ith_antbox, qpos, qvel):
        box_x_positions = np.random.uniform(-6, 6, size=(2,))
        pos_y_joint, pos_z_joint  = self.get_box_joint_pos(ith_antbox)
        vel_y_joint, vel_z_joint = self.get_box_joint_vel(ith_antbox)
        if ith_antbox == 0:
            box_x_position = box_x_positions[0]
        elif ith_antbox ==1:
            box_x_position = box_x_positions[1]
        else:
            box_x_position = 0# -20
            raise NameError('Wrong box number')

        self.box_pos[ith_antbox] = box_x_position
        self.model.body_pos[self.model.body_name2id("box" + str(ith_antbox))][0] = box_x_position
        qpos[pos_y_joint] = self._init_box_y_position + last_goal_position
        qpos[pos_z_joint] = self._box_z_position

        qvel[vel_y_joint] = 0
        qvel[vel_z_joint] = 0
        return qpos, qvel

    def get_box_joint_pos(self, ith_antbox):

        joint_OBJTy = self.sim.model.get_joint_qpos_addr('OBJTy'+str(ith_antbox))
        joint_OBJTz = self.sim.model.get_joint_qpos_addr('OBJTz'+str(ith_antbox))
        return joint_OBJTy, joint_OBJTz
        #return joint_OBJTx, joint_OBJTy, joint_OBJTz

    def get_box_joint_vel(self, ith_antbox):
        # joint_OBJTx = self.sim.model.get_joint_qvel_addr('OBJTx')
        joint_OBJTy = self.sim.model.get_joint_qvel_addr('OBJTy'+str(ith_antbox))
        joint_OBJTz = self.sim.model.get_joint_qvel_addr('OBJTz'+str(ith_antbox))
        return joint_OBJTy, joint_OBJTz
        #return joint_OBJTx, joint_OBJTy, joint_OBJTz

    def wind_force(self, force):
        # torso_index = self.sim.model._body_name2id["agent_torso"]
        # self.sim.data.xfrc_applied[torso_index][0] = force
        pass


    def distance_to_coins(self):

        coin_task_no = int(self._task_sets[self.task_order[self.subtaskid]][-1])
        if coin_task_no >= 5:
            coin_task_no = 4
        firstcoin_name = "coin_geom1_" + str(coin_task_no)
        secondcoin_name = "coin_geom2_" + str(coin_task_no)

        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]

        idx1 = self.model.geom_name2id(firstcoin_name)
        idx2 = self.model.geom_name2id(secondcoin_name)
        first_coin_x, first_coin_y = self.sim.model.geom_pos[idx1][0], self.sim.model.geom_pos[idx1][1]

        second_coin_x, second_coin_y = self.sim.model.geom_pos[idx2][0], self.sim.model.geom_pos[idx2][1]
        first_coin_distance = math.sqrt((agent_x - first_coin_x) ** 2 + (agent_y - first_coin_y) ** 2)
        second_coin_distance = math.sqrt((agent_x - second_coin_x) ** 2 + (agent_y - second_coin_y) ** 2)
        return first_coin_distance, second_coin_distance, coin_task_no

    def put_firstCoin_away(self, coin_task_no):
        firstcoin_name = "coin_geom1_" + str(coin_task_no)
        idx = self.model.geom_name2id(firstcoin_name)
        self.sim.model.geom_pos[idx][0] = 0
        self.sim.model.geom_pos[idx][1] = -20

    def put_secondCoin_away(self, coin_task_no):
        secondcoin_name = "coin_geom2_" + str(coin_task_no)
        idx = self.model.geom_name2id(secondcoin_name)
        self.sim.model.geom_pos[idx][0] = 0
        self.sim.model.geom_pos[idx][1] = -22

    def _set_curbs_xposition(self, idxleft, idxright, ith_antgoal):
        x_pos_samplers = np.random.uniform(0, 1, size=(2,))
        if ith_antgoal == 0:
            x_pos_sampler = x_pos_samplers[0]
        elif ith_antgoal == 1:
            x_pos_sampler = x_pos_samplers[1]
        else:
            x_pos_sampler = 0.5
        #x_pos_sampler = np.random.uniform(0.3, 0.7)
        x_pos = (x_pos_sampler * 0.8 + 0.1) * 20 - 10
        self.corridor_pos[ith_antgoal] = x_pos

        right_curb_leftend_pos = x_pos + self.corridor_width / 2
        right_curb_length = 10 - right_curb_leftend_pos
        right_curb_pos = right_curb_leftend_pos + right_curb_length / 2

        self.sim.model.geom_pos[idxright][0] = right_curb_pos
        self.sim.model.geom_size[idxright][0] = right_curb_length / 2

        left_curb_rightend_pos = x_pos - self.corridor_width / 2
        left_curb_length = left_curb_rightend_pos + 10
        left_curb_pos = left_curb_rightend_pos - left_curb_length / 2

        self.sim.model.geom_pos[idxleft][0] = left_curb_pos
        self.sim.model.geom_size[idxleft][0] = left_curb_length / 2
        # print("x_pos is at:", self.x_pos, "right curb pos", right_curb_pos, "left curb pos", left_curb_pos)

    def distance_to_goal(self, goal_y):
        goal_x = 0
        goal_y = goal_y
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        #distance = math.sqrt((goal_x - agent_x) ** 2 + (goal_y - agent_y) ** 2)
        distance = goal_y - agent_y
        return distance

    def distance_to_door(self, ith, offset):
        if ith >1:
            ith = 1
        door_x = self.corridor_pos[ith]
        door_y = offset
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        distance =  math.sqrt((door_x - agent_x) **2 + (door_y - agent_y) **2)
        return distance

    def box_distance_to_ant(self, ith_antbox, offset):
        if ith_antbox >=2:
            ith_antbox = 1
        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]

        distance = math.sqrt((agent_x - self.box_pos[ith_antbox]) **2 + (agent_y - offset + 6) **2)
        return distance

    def get_all_task_idx(self):
        return range(1)

    def reset_task(self, idx):
        self.reset()

    def distance_to_cliff_end(self, ith_antbox, offset):
        if ith_antbox >=2:
            ith_antbox = 1
        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]
        cliff_center_x =  self.box_pos[ith_antbox]
        cliff_center_y = offset + 4.5
        distance = math.sqrt((agent_x - cliff_center_x) **2 + (agent_y - cliff_center_y) **2)
        return distance