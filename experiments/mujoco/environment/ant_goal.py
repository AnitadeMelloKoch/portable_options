import os
import math

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import matplotlib.pyplot as plt


class AntGoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=1, env_type='train', randomize_tasks=True, eval=eval):

        self._task = task
        self.env_type = env_type
        self.tasks = self.sample_tasks(n_tasks)
        self.eval = eval
        #these will get overiden when we call reset_task from the outside.
        self._x_pos_sampler = 0.5
        self._curb_y_pos = 10
        self.random_steps = 5
        self.max_step = 400
        self.passing_reward = 10
        self.goal_reward = 20
        self.outside_reward = -5
        self.floor_width = 10
        self.corridor_width = 3
        self.colliding_reward = 0
        self.survive_reward = 0
        self.distanceToDoorRewardWeight = 20
        self.distanceToDoorGoalWeight = 20
        self.current_step = 0
        self.x_pos = (self._x_pos_sampler *0.8 + 0.1)* 20 - 10
        self._outside = False
        self.colliding_with_curb = False
        self._passingDoor = False
        self.ob_shape = {"joint": [29]}
        self.ob_type = self.ob_shape.keys()
        xml_path = os.path.join(os.getcwd(), "portable/policy/envs/assets/ant-goal.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)


    def step(self, a, skillpolicy=None, id=0):
        #do the simulation and check how much we moved on the y axis
        yposbefore = self.get_body_com("agent_torso")[1]
        distanceToDoorBefore = self.distance_to_door()
        distanceToGoalBefore = self.distance_to_goal()
        self.do_simulation(a, self.frame_skip)
        #temp stuff
        #self.render()
        #increment 1 step
        self.current_step += 1
        yposafter = self.get_body_com("agent_torso")[1]
        agent_xpos = self.get_body_com("agent_torso")[0]
        # forward_reward = (yposafter - yposbefore) / self.dt

        #if collide with the wall or went outside, then we shall stop and give agent a big penalty.
        outside_reward = 0
        if agent_xpos<-self.floor_width or agent_xpos >=self.floor_width:
            self._outside = True
            outside_reward = self.outside_reward

        #we are currently not using colliding reward and collision detection
        colliding_reward = 0
        if self.collision_detection("curbleft") or self.collision_detection("curbright"):
            self.colliding_with_curb = False
            colliding_reward = self.colliding_reward

        #check if the agent got tipped over
        tipped_over = self.get_body_com("agent_torso")[2]<=0.3 # or self.get_body_com("agent_torso")[2]>=1


        # if we haven't passed the door, then we can get reward when pass the door.
        passing_reward = 0
        if yposafter >= self.sim.data.get_geom_xpos('curbleft')[1] and not self._passingDoor:
            self._passingDoor = True
            passing_reward = self.passing_reward

        distanceToDoorAfter = self.distance_to_door()
        distanceToDoorReward = 0
        if not self._passingDoor:
            distanceDifference = distanceToDoorBefore-distanceToDoorAfter
            distanceToDoorReward = distanceDifference*self.distanceToDoorRewardWeight


        distanceToGoalAfter = self.distance_to_goal()
        distanceToGoalReward = 0
        if self._passingDoor:
            distanceDifference = distanceToGoalBefore - distanceToGoalAfter
            distanceToGoalReward = distanceDifference * self.distanceToDoorGoalWeight


        #control cost for the agent, I don't think we need it because the ball will just move, leave it for now with a smaller weight.
        ctrl_cost = 0 #0.5 * np.square(a).sum()



        state = self.state_vector()
        #check when we need to finish the current episode
        done = False
        if self._outside or self.current_step >= self.max_step or tipped_over:
            done = True
        ob = self._get_obs()
        goal_reward = 0
        success = False
        # cam = self.render(mode="rgb_array", width=96, height=96, camera_name="track")

        if distanceToGoalAfter < 0.1:
            done = True
            success = True
            goal_reward = self.goal_reward
        reward =  outside_reward + passing_reward + goal_reward + distanceToDoorReward + distanceToGoalReward - ctrl_cost
        #print("contact cost:", contact_cost, 'ctrl_cost', ctrl_cost, "passing_reward", passing_reward, "goal_reward", goal_reward,"distanceToDoorReward",distanceToDoorReward, "distanceToGoalReward", distanceToGoalReward)
        return (
            ob,
            reward,
            done,
            dict(
                reward_ctrl=-ctrl_cost,
                success = success,
                obs_img = self._get_img_obs() if self.eval else self._get_obs(),
                position = (self.sim.data.qpos.flat[0], self.sim.data.qpos.flat[1])
            ),
        )
    #the new observation is [agent position, curb1 y axis position, agent velocity]

    def _get_obs(self):
        return np.concatenate(
            # x:0 y:1
            [
                #self.sim.data.qpos.flat[0], #x position
                #np.array([self.sim.data.get_geom_xpos('curbleft')[1] - self.sim.data.get_body_xpos('agent_torso')[1]]), #relative y position
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:],

            ]
        )

    def _get_img_obs(self):

        cam = self.render(mode="rgb_array", width=128, height=128, camera_name="track")

        return cam

    def reset_model(self):
        self.current_step = 0
        self._outside = False
        self._passingDoor = False
        self.colliding_with_curb = False
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self._put_curbs()
        self.set_state(qpos, qvel)

        for _ in range(int(self.random_steps)):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        cam = self.render(mode="rgb_array", width=96, height=96, camera_name="track")

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
        self._put_curbs()
        qpos = self.init_qpos
        # random position if not specified
        if pos is None:
            pos = (
                self.np_random.uniform(-9, 9, size=1),  # x 
                self.np_random.uniform(-10, 25, size=1)  # y
            )

        qpos[0] = pos[0]
        qpos[1] = pos[1]
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        # random steps to ensure proper dynamic 
        for _ in range(5):
            self.step(self.unwrapped.action_space.sample())

        x, y, z = self.get_body_com("agent_torso")
        return x, y

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _put_curbs(self):
        idxleft = self.model.geom_name2id('curbleft')
        idxright = self.model.geom_name2id('curbright')


        right_curb_leftend_pos = self.x_pos + self.corridor_width / 2
        right_curb_length = 10 - right_curb_leftend_pos
        right_curb_pos = right_curb_leftend_pos + right_curb_length / 2


        self.sim.model.geom_pos[idxright][0] =   right_curb_pos
        self.sim.model.geom_pos[idxright][1] = self._curb_y_pos
        self.sim.model.geom_size[idxright][0] =  right_curb_length / 2

        left_curb_rightend_pos = self.x_pos - self.corridor_width / 2
        left_curb_length = left_curb_rightend_pos + 10
        left_curb_pos = left_curb_rightend_pos - left_curb_length / 2

        self.sim.model.geom_pos[idxleft][0] =  left_curb_pos
        self.sim.model.geom_pos[idxleft][1] = self._curb_y_pos
        self.sim.model.geom_size[idxleft][0] = left_curb_length / 2
        #print("x_pos is at:", self.x_pos, "right curb pos", right_curb_pos, "left curb pos", left_curb_pos)


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

    def distance_to_door(self):
        door_x = self.x_pos
        door_y = self.sim.data.get_geom_xpos('curbleft')[1]
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        distance =  math.sqrt((door_x - agent_x) **2 + (door_y - agent_y) **2)
        return distance

    def distance_to_goal(self):
        goal_x = 0 #self.sim.data.get_geom_xpos('goal')[0]
        goal_y = 25 #self.sim.data.get_geom_xpos('goal')[1]
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        #distance =  math.sqrt((goal_x - agent_x) **2 + (goal_y - agent_y) **2)
        distance = goal_y - agent_y
        return distance

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def sample_tasks(self, num_tasks):

        if self.env_type == 'test':
            x_pos_samplers = np.linspace(0.1, 0.9, num_tasks)
            curb_y_positions = np.linspace(9.9, 10.1, num_tasks)
            tasks = [{'x_pos_sampler': x_pos_sampler, 'curb_y_pos': curb_y_pos} for x_pos_sampler, curb_y_pos in zip(x_pos_samplers, curb_y_positions)]
        else:
            x_pos_samplers = np.random.uniform(0, 1, size=(num_tasks,))
            curb_y_positions = np.random.uniform(9.99999, 10.00001, size=(num_tasks,))
            tasks = [{'x_pos_sampler': x_pos_sampler, 'curb_y_pos': curb_y_pos} for x_pos_sampler, curb_y_pos in zip(x_pos_samplers, curb_y_positions)]
        return tasks



    def reset_task(self, idx):


        self._task = self.tasks[idx]
        self._x_pos_sampler =  self._task['x_pos_sampler']
        self._curb_y_pos = self._task['curb_y_pos']
        self.x_pos = (self._x_pos_sampler *0.8 + 0.1 )* 20 - 10


        self.reset()
