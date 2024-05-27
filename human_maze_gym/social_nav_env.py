import warnings

import colorsys
import gym
import matplotlib.pyplot as plt
from gym import spaces
from torch.utils.data import DataLoader

from human_maze_gym.trajectories import TrajectoryDataset, Collate
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
import d3rlpy
plt.rc('figure', figsize=(10, 10))
from tqdm import tqdm
from human_maze_gym.utils import *




class SocialNavEnv(gym.Env):

    def __init__(self, device, action_norm=False, use_robot_model=False,
                 test_mode=False, simple_state=False, XYAction=True, max_speed_norm=1.5, bc=False):
        # env params, do not change!
        self.use_robot_model = use_robot_model
        if self.use_robot_model:
            print('Using robot kinematic constrains model')
        self.XYAction = XYAction
        np.random.seed(seed=43)
        self.simple_state = simple_state
       # self.scene_mode = scene_mode
        self.device = device
        self.goal_thresh = 0.2
        self.bc = bc  # we need next observation for behaviour cloning
        self.test_data_flag = test_mode
        self.dataset_name = 'univ-big'  # which dataset should be used: Social-GAN (univ) or Trajnet + Social-GAN (univ_big)
  
        self.agent_hist = 7
        self.start_time = self.agent_hist
        self.collision_dist = 0.2
        self.dt = 0.4  # Don't change! Discretization of the dataset
        # robot model param for DWC
        self.robot_params_dict = {}
        self.robot_params_dict['use_robot_model'] = self.use_robot_model # Flag for robot param constrains usage
        self.robot_params_dict['max_speed'] = .7  # [m/s] # 0.5 locobot
        self.robot_params_dict['min_speed'] = 0.  # [m/s]
        self.robot_params_dict['max_yaw_rate'] = 1.0  # [rad/s]
        self.robot_params_dict['max_accel'] = .5  # [m/ss]
        self.robot_params_dict['max_delta_yaw_rate'] = 3.2  # [rad/ss]
        self.robot_params_dict['collision_dist'] = self.collision_dist  # [m]

        # percetoption param
        self.prox_rad = 5  # In what radius around our robot do we keep track of human agents?
        self.num_agents = 5  # How many agents can we keep track of / include in our state?
      
        ############ Reinforcment Learning Parameteres ############
        self.action_norm = action_norm
        self.min_x = -max_speed_norm  # based on dataset analysis 1.5 is max speed observed. However SAC learns to always move at 1.5 speed so we reduce it to 0.5 for more realistic movement
        self.min_y = -max_speed_norm
        self.max_x = max_speed_norm
        self.max_y = max_speed_norm

        # assert not (self.action_norm and not XYAction), 'action norm only for holonomic XY actions'
        # Setup observation and action spaces
        if self.simple_state:
            n = 19
        elif not self.bc:
            n = int( 2 * (self.num_agents+1) * (self.agent_hist + 1) + (self.num_agents + 1) ** 2 + 5 + 2) 
        else:
            n = int( 2 * (self.num_agents+1) * (self.agent_hist + 2) + (self.num_agents + 1) ** 2 + 5 + 2) # deprecated not in use

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[n])
        self.action_space = spaces.Box(np.float32(-np.ones(2)),np.float32(np.ones(2)))
        # Reward function parameters
        self.alpha_sig = -1.5 # -1.5
        self.beta = 0.5  
        self.goal_reward = 20.
        self.collision_penalty = -25.
        self.time_pen = -77. #-2.


        self.get_data_loader()

        # Reset simulation to populate env data
        _ = self.reset()

        # We have 3 test scenarios
        # 1: The robot goal ist set to a minimum distance of 3(m) distance within 12 timesteps
        # 2: The robot goal ist set to a minimum distance of 8(m) distance within 50 timesteps
        # 3: The robot goal ist set to a minimum distance of 3(m) distance within 50 timesteps

    def set_scene_mode_short(self,):
        self.min_goal_range = 3. # Set the min goal length from the initial robot position (m)
        self.bonus_time_steps = 10
        self.human_future = 12
        self.goal_thresh = 0.5
        self.episode_time = self.agent_hist + self.human_future + self.bonus_time_steps # How long should the episodes last before being terminated?

    def set_scene_mode_long(self,):
        self.min_goal_range = 8.
        self.bonus_time_steps = 20
        self.human_future = 50
        self.goal_thresh = 0.5
        self.episode_time = self.agent_hist + self.human_future + self.bonus_time_steps

    def set_scene_mode_long_train(self,):
        self.min_goal_range = 3.
        self.bonus_time_steps = 20
        self.human_future = 50
        self.goal_thresh = 0.5
        self.episode_time = self.agent_hist + self.human_future + self.bonus_time_steps


    def get_data_loader(self,):

        self.set_scene_mode_long()


        if self.test_data_flag:
                path = get_dset_path('test', self.dataset_name =='univ')
                augment = False
        else:
            path = get_dset_path('train', self.dataset_name =='univ')
            augment = True
        print(f'Using {self.dataset_name} dataset')

        self.reset_counter = 0
        dset = TrajectoryDataset(
                path,
                obs_len=self.agent_hist,
                pred_len=self.human_future,
                episode_time = self.episode_time,
                skip=1,
                min_goal_range=self.min_goal_range,
                delim='tab')

        collate = Collate(augment, 'data', pred_len=self.human_future,
                        obs_len=self.agent_hist, episode_length=self.episode_time)
        self.loader = DataLoader(
            dset,
            batch_size=1,
            shuffle=False if self.test_data_flag else True,
            num_workers=0,
            collate_fn=collate.seq_collate,
            pin_memory=False)

        self.loader_iter = iter(self.loader)
        self.num_seqs = dset.num_seq


        if not self.test_data_flag:
            self.data_loader_picker_flag = True
            self.reset_counter_short = 0

            self.set_scene_mode_short()

            dset_short = TrajectoryDataset(
            path,
            obs_len=self.agent_hist,
            pred_len=self.human_future,
            episode_time = self.episode_time,
            skip=1,
            min_goal_range=self.min_goal_range,
            delim='tab')

            collate_short = Collate(augment, 'data', pred_len=self.human_future,
                            obs_len=self.agent_hist, episode_length=self.episode_time)
            self.loader_short = DataLoader(
                dset_short,
                batch_size=1,
                shuffle=False if self.test_data_flag else True,
                num_workers=0,
                collate_fn=collate_short.seq_collate,
                pin_memory=False)

            self.loader_iter_short = iter(self.loader_short)
            self.num_seqs_short = dset_short.num_seq
            self.set_scene_mode_long_train()



    def min_max_norm(self, input):
        r"""Min-Max normalization
            Input will be normalized in range ``[-1.0, 1.0]``.
            .. math::
                a' = (a - \min{a}) / (\max{a} - \min{a}) * 2 - 1
        """
        np_min = np.array([[self.min_x, self.min_y]])
        np_max = np.array([[self.max_x, self.max_y]])
        test = (input - np_min) / (np_max - np_min) * 2 - 1
        return test
    
    def get_agent_states(self, time, filter_p=True):
        '''
        Return a sorted list of agent poses and histories, where the closest
        agent appears first. Only agents within self.prox_rad are included
        '''
        # time = time + 1
        neigh_filtered_padded = np.zeros([self.agent_hist + 2, self.num_agents,2])#* -np.inf
        neight_matrix_padded = np.zeros([self.num_agents+1, self.num_agents+1])  # * -np.inf
        t_nei_idx = self.nei_index_t_no_robotHuman[time].sum(axis=-1)
        t_nei_idx = t_nei_idx > 0
        if t_nei_idx.sum() == 0: # special case for number_agent = 2, if we delete one agent here the other has no neighbour so it will be not considered
            t_nei_idx = self.nei_index_t[time].sum(axis=-1)
            t_nei_idx = t_nei_idx > 0
            t_nei_idx[self.robot_idx] = False
        robot_neigh = self.hole_traj[(time-self.agent_hist) : time + 2, t_nei_idx]
        assert not ((robot_neigh[-2].sum() == 0).any() and self.nei_index_t[time].sum() != 0)
        if filter_p:
            d = np.linalg.norm(robot_neigh[-2] - self.robot_state[:2], axis=1)
            neig_in_proxrad_idx = np.argwhere(d <= self.prox_rad)[:,0]
            sorted_neigh_indx = np.argsort(d)[:self.num_agents]
            neigh_idx_filtered = np.intersect1d(sorted_neigh_indx, neig_in_proxrad_idx)
            neigh_filtered = robot_neigh[:, neigh_idx_filtered]
            neigh_filtered_padded[:,0:len(neigh_idx_filtered)] = neigh_filtered
            neight_matrix = np.ones((len(neigh_idx_filtered) + 1, len(neigh_idx_filtered) + 1), int)  # +1 for robot
            np.fill_diagonal(neight_matrix, 0)
            neight_matrix_padded[0:len(neigh_idx_filtered) + 1, 0:len(neigh_idx_filtered) + 1] = neight_matrix
            return neigh_filtered_padded, neight_matrix_padded
        else:
            return robot_neigh[:-1], self.ids_plot[t_nei_idx]

    def reset(self):
        '''
        Reset the entire simulation, including new dataset, start and goal
        '''
        # Set new
        if not self.test_data_flag:
          #  choice = np.random.choice(1, 1, replace=False)
           # if choice:
            if self.data_loader_picker_flag:
                self.set_scene_mode_long_train()
                batch = next(self.loader_iter)
                state = self.init_scene(batch=batch, data_loader_picker_flag=self.data_loader_picker_flag)
                self.reset_counter += 1
                if self.num_seqs == self.reset_counter:
                    print('Counter reset at: ' + str(self.reset_counter))
                    self.data_loader_picker_flag = False
                    self.reset_counter = 0
                    self.loader_iter = iter(self.loader)
            else:
                self.set_scene_mode_short() 
                batch = next(self.loader_iter_short)
                state = self.init_scene(batch=batch)
                self.reset_counter_short += 1
                # print(self.reset_counter)
                if self.num_seqs_short == self.reset_counter_short:
                    print('Counter reset at: ' + str(self.reset_counter_short))
                    self.data_loader_picker_flag = True
                    self.reset_counter_short = 0
                    self.loader_iter_short = iter(self.loader_short)
            return state, 'Place_Holder'

        else:
            batch = next(self.loader_iter)
            state = self.init_scene(batch=batch)
            self.reset_counter += 1
            # print(self.reset_counter)
            if self.num_seqs == self.reset_counter:
                print('Counter reset at: ' + str(self.reset_counter))
                self.reset_counter = 0
                self.loader_iter = iter(self.loader)
            return state, 'Place_Holder'

    def actionXYtoROT(self, actionXY, robot_state):
        # robot_state state[v_x(m / s), v_y(m / s), yaw(rad), v(m / s), omega(rad / s)] x and y are displacments to last state
        v, yaw = self.cart2pol(actionXY[0], actionXY[1])
        last_yaw = robot_state[2]
        diff = np.arctan2(np.sin(yaw - last_yaw), np.cos(yaw - last_yaw))
        omega_t = (diff) / self.dt
        v_t = v / self.dt
        return np.array([v_t, omega_t])

    def step(self, proposed_action, coll_done=True, human=False):
        '''
        Main function called at each timestep. Given some action, in self.actions
        return our new observed state, reward given and whether we've terminated
        proposed_action
        '''
        # robot_state state[x(m), y(m), v_x(m / s), v_y(m / s), yaw(rad), v(m / s), omega(rad / s)] x and y are displacments to last state
        info = ''
        prev_pose = np.copy(self.robot_state)
        prev_robot_pos = np.array([(prev_pose[0]), (prev_pose[1])])
        if self.action_norm and not human and self.XYAction:
            proposed_action = proposed_action * self.max_x  # is norm. transfer input of [-1,1] to max x and y (should be the same)

        if self.XYAction or human:
            u = self.actionXYtoROT(proposed_action, self.robot_state[2:])
        else:
            if self.action_norm:
                u = proposed_action * np.array([self.robot_params_dict['max_speed'],self.robot_params_dict['max_yaw_rate'] ])
            else:
                u = proposed_action

        if self.robot_params_dict['use_robot_model'] and not human:
            new_robot_state = dynamic_window(torch.from_numpy(self.robot_state[2:]).unsqueeze(0),
                                             torch.from_numpy(u).unsqueeze(0),
                                             self.robot_params_dict, self.dt).numpy()[0]
            action = new_robot_state[0:2] * self.dt
            self.robot_state[2:] = new_robot_state
        else:
            action = proposed_action
            theta = np.arctan2(proposed_action[1], proposed_action[0])
            self.robot_state[2:4] = action / self.dt
            self.robot_state[4] = theta
            self.robot_state[5:] = u

        self.action = action # for correct data gathering in dataset
        self.robot_state[:2] += self.action
        # Update pose history for rendering
        self.pose_history = np.concatenate((self.pose_history, np.expand_dims(self.robot_state[0:2], axis=0)), axis=0)
        self.pose_history = self.pose_history[1:]

        # Update observation
        self.time += 1
        agents_states, neight_matrix = self.get_agent_states(self.time)

        if self.simple_state:
            state = np.concatenate((self.robot_state,
                                    (agents_states[-2] - self.pose_history[-1]).flatten(),
                                    self.goal - self.pose_history[-1] ))
        else:
            curr_scene_obs = np.concatenate([np.expand_dims(self.pose_history, axis=1), agents_states[:-1]], axis=1 )
            curr_scene_obs_mask = np.where(curr_scene_obs != 0, True, False)
            curr_scene_obs = (curr_scene_obs - self.robot_state[:2]) * curr_scene_obs_mask
        #    print(curr_scene_obs[-1])
            state = np.concatenate((curr_scene_obs.flatten(), neight_matrix.flatten(), self.robot_state[2:], self.goal - self.robot_state[:2]))

        #Reset termination flags
        done = False
        reward = 0.0
        
        current_scene = np.concatenate((np.expand_dims(self.robot_state[0:2], axis=0), agents_states[-2]), axis = 0)
        num_coll, dist_matrix = fast_coll_counter(torch.from_numpy(current_scene), robot_id=0, coll_dist = self.collision_dist)#

        #Distance Reward Function
        reward = self.beta * ( 1 - (np.linalg.norm(self.robot_state[0:2] - self.goal) / np.linalg.norm(self.start - self.goal)))
    
        #Sigmoid Risk Reward Function
        sig_risk_comp = torch.where(dist_matrix[1:]<=6.0, self.alpha_sig * (1 - torch.sigmoid((dist_matrix[1:] - 0.4)*6.8)) + 0.2, 0.0)
        reward += float(torch.min(sig_risk_comp))


        #Collisions
        if num_coll > 0:
            reward = self.collision_penalty
            info = 'col'
            if coll_done:
                done = True
                return state, reward, done, False, info

        # Did we reach the goal?
        if np.linalg.norm(self.robot_state[0:2] - self.goal) < self.goal_thresh:
            # print('GOAAAAL')
            done = True
            reward = self.goal_reward
            if info != 'col':
                info = 'goal'
            return state, reward, done, False, info

        # Are we out of time?
        if self.time + 3 >= self.episode_time:
            done = True
            if info != 'col':
                reward = self.time_pen
                info = 'time_out'
            return state, reward, done, False, info

        return state, reward, done, False, info

    def render(self):
        '''
        Render the current timestep using pyplot
        '''
        current_neigh, current_ids_plot = self.get_agent_states(self.time,filter_p=False)
        a = self.hole_traj[:, :, 0].flatten()
        b = self.hole_traj[:, :, 1].flatten()
        self.maxx, self.minx = np.max(self.hole_traj[:, :, 0]), \
                               np.min(a[a != 0])
        self.maxy, self.miny = np.max(self.hole_traj[:, :, 1]), \
                               np.min(b[b != 0])

        plt.rcParams['axes.facecolor'] = (0.9, 0.9, 0.9)
        ######
        plt.clf()
        # plt.figure(figsize=(10, 10))
        plt.gca().set_aspect("equal")
        # plt.axis([-10, 20 + 1, -10, 20 + 1])
        plt.axis([self.minx-3, self.maxx+3, self.miny-3, self.maxy+3])

        for i in range(0, current_neigh.shape[1]):
            ag = current_neigh[:,i]
            color = tuple(self.colors[current_ids_plot[i]])
            a_pos = plt.Circle(ag[-1], 0.2 / 2., fill=True, color=color,
                               label='Agents')
            plt.gca().add_patch(a_pos)
            for t in range(0, self.agent_hist):
                if ag[t].sum() != 0.:    # history can be padded with zero
                    his_pos = plt.Circle(ag[t], 0.2 / 2., fill=False,
                                         color=color, label='Agents_hist')
                    plt.gca().add_patch(his_pos)

        plt.plot([self.start[0],self.goal[0]],[self.start[1],self.goal[1]],color=(0,1,0),zorder=0)
        rob_pos = plt.Circle(self.robot_state, 0.2 / 2., fill=True, color=(0, 0, 0),
                             label='Robot')
        replaced_human = self.hole_traj[self.time, self.robot_idx][0]
        rob_range = plt.Circle(self.robot_state, self.prox_rad, fill=False, color='r',
                               label='range')
        l = plt.Line2D((self.pose_history[- 2, 0], self.pose_history[- 1, 0]),
                          (self.pose_history[- 2, 1], self.pose_history[- 1, 1]), color='r', ls='solid')
        plt.plot(self.goal[0], self.goal[1], marker='X', color='black')
        plt.plot(replaced_human[0], replaced_human[1], marker='*', color='black')
        plt.gca().add_patch(rob_pos)
        plt.gca().add_line(l)
        plt.gca().add_patch(rob_range)

        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.pause(0.1)

    def save_render(self):
        '''
        Render the current timestep using pyplot
        '''
        current_neigh, current_ids_plot = self.get_agent_states(self.time,filter_p=False)
        a = self.hole_traj[:, :, 0].flatten()
        b = self.hole_traj[:, :, 1].flatten()
        self.maxx, self.minx = np.max(self.hole_traj[:, :, 0]), \
                               np.min(a[a != 0])
        self.maxy, self.miny = np.max(self.hole_traj[:, :, 1]), \
                               np.min(b[b != 0])

        plt.clf()
        plt.gca().set_aspect("equal")

        plt.axis([self.robot_state[0] - 6, self.robot_state[0] + 6, self.robot_state[1] - 6, self.robot_state[1] + 6])
        for i in range(0, current_neigh.shape[1]):
            ag = current_neigh[:,i]
            color = tuple(self.colors[current_ids_plot[i]])
            a_pos = plt.Circle(ag[-1], 0.2 / 2., fill=True, color=color,
                               )
            plt.gca().add_patch(a_pos)
            for t in range(0, self.agent_hist):
                if ag[t].sum() != 0.:    # history can be padded with zero
                    his_pos = plt.Circle(ag[t], 0.2 / 2., fill=False,
                                         color=color)
                    plt.gca().add_patch(his_pos)
        for t in range(0, self.agent_hist):
            if self.robot_traj[t].sum() != 0.:  # history can be padded with zero
                his_pos = plt.Circle(self.pose_history[t], 0.2 / 2., fill=False,
                                     color=(0, 0, 0), )
                plt.gca().add_patch(his_pos)

        plt.plot(current_neigh[-1,-1, 0], current_neigh[-1, -1, 1], marker='o', color='cyan', markersize=10,
                 label='Humans Position')

        plt.plot([self.start[0],self.goal[0]],[self.start[1],self.goal[1]],color=(0,1,0),zorder=0, label='Start2Goal')
        rob_pos = plt.Circle(self.robot_state, 0.2 / 2., fill=True, color=(0, 0, 0))
        plt.plot(self.robot_state[0], self.robot_state[1], marker='o', color='black', markersize=10, label='Robot Position')
        replaced_human = self.hole_traj[self.time, self.robot_idx][0]
        l = plt.Line2D((self.pose_history[- 2, 0], self.pose_history[- 1, 0]),
                          (self.pose_history[- 2, 1], self.pose_history[- 1, 1]), color='r', ls='solid')
        plt.plot(self.goal[0], self.goal[1], marker='X', color='black', markersize=20, label='Goal')
        plt.plot(replaced_human[0], replaced_human[1], marker='*', color='black', markersize=10, label='Selected Human')
        plt.gca().add_patch(rob_pos)
        plt.gca().add_line(l)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(True)
        ax.axes.yaxis.set_visible(True)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.pause(0.1)

        if self.time > 18:
            ax.legend(loc=3)
            plt.savefig(f'./img/img_{self.time}.png',
                        transparent=False,
                        facecolor='white',
                        bbox_inches='tight'
                        )




    @staticmethod
    def get_spaced_colors(n):
        '''
        Helper function that assigns somewhat unique colors to agents
        '''
        colors = np.zeros((n,3))
        idxs = np.arange(0,n,1).astype(int)
        np.random.shuffle(idxs)
        j = 0
        for i in idxs:
            h = j*1.0/n
            rgb = colorsys.hsv_to_rgb(h,1,1)
            colors[i] = rgb
            j+=1
        return colors

    def cart2pol(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi
    
    def smallest_angle_degs(self, angle1, angle2):
        """ Takes two angles ranging between 0-360"""
        angle_diff = torch.abs(angle1-angle2)
        smallest_angle = torch.minimum(angle_diff, 360-angle_diff)

        return smallest_angle

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def init_scene(self, batch, data_loader_picker_flag=False):
        '''
        Init the scene here with the offline data
        '''
        self.obs_traj, self.pred_traj_gt, self.obs_traj_rel, self.pred_traj_gt_rel \
            , val_mask, _, _, self.nei_index_t, self.idx_valid = batch
        self.colors = self.get_spaced_colors(self.obs_traj.shape[1])
        self.time = self.start_time
        self.robot_idx = self.idx_valid # np.random.choice(self.idx_valid, 1)
        self.nei_index_t_no_robotHuman = np.copy(self.nei_index_t)
        self.nei_index_t_no_robotHuman[:, self.robot_idx, :] = 0
        self.nei_index_t_no_robotHuman[:, :, self.robot_idx] = 0
        self.hole_traj = np.concatenate([self.obs_traj,
                                         self.pred_traj_gt], axis=0)
        self.ids_plot = np.array(range(self.obs_traj.shape[1])) # ids of tracks for rendering
        self.hole_traj_rel = np.concatenate([self.obs_traj_rel,
                                             self.pred_traj_gt_rel], axis=0)
        self.robot_traj = self.hole_traj[:, self.robot_idx][:, 0]
        self.robot_traj_rel = self.hole_traj_rel[:, self.robot_idx][:, 0]
        org_max_scene_goal = self.agent_hist + self.human_future - 1
        stoch_max_scence_goal = np.random.randint(int(org_max_scene_goal/2), org_max_scene_goal+1)
        self.goal = self.robot_traj[stoch_max_scence_goal] if data_loader_picker_flag else self.robot_traj[org_max_scene_goal] # robot goal in global space 
        assert self.goal.sum() != 0
        self.start = self.robot_traj[self.time]
        yaw_test = np.arctan2(self.robot_traj_rel[self.start_time, 1], self.robot_traj_rel[self.start_time, 0]) # y/x

        # --------------start: robot state init  ------------------------
        _, last_yaw = self.cart2pol(self.robot_traj_rel[self.start_time-1, 0],
                               self.robot_traj_rel[self.start_time-1, 1])
        v, yaw = self.cart2pol(self.robot_traj_rel[self.start_time, 0],
                            self.robot_traj_rel[self.start_time, 1])
        v_t = v / self.dt
        diff = np.arctan2(np.sin(yaw - last_yaw), np.cos(yaw - last_yaw))
        omega_t = (diff) / self.dt
        if self.robot_params_dict['use_robot_model']:
            v_t = np.clip(v_t, a_min=self.robot_params_dict['min_speed'],
                              a_max=self.robot_params_dict['max_speed'])
            omega_t = np.clip(omega_t, a_min=-self.robot_params_dict['max_yaw_rate'],
                                  a_max=self.robot_params_dict['max_yaw_rate'])
            v_x, v_y = self.pol2cart(v_t, omega_t)
        else:
            v_x, v_y = self.robot_traj_rel[self.start_time, 0] / self.dt, self.robot_traj_rel[self.start_time, 1] / self.dt
        # robot_state state[x(m), y(m), v_x(m / s), v_y(m / s), yaw(rad), v(m / s), omega(rad / s)] x and y are displacments to last state
        self.robot_state = np.array([self.start[0], self.start[1], v_x, v_y, yaw, v_t, omega_t])
        # --------------end: robot state init  ------------------------

        self.pose_history = self.robot_traj[:self.agent_hist + 1]
        agents_states, neight_matrix = self.get_agent_states(self.time)

        if self.simple_state:
            state = np.concatenate((self.robot_state[2:],
                                    (agents_states[-1] - self.pose_history[-1]).flatten(),
                                    self.goal - self.pose_history[-2] ))
        else:
            curr_scene_obs = np.concatenate([np.expand_dims(self.pose_history, axis=1), agents_states[:-1]], axis=1 ) #- self.robot_state[:2]
            curr_scene_obs_mask = np.where(curr_scene_obs != 0, True, False)
            curr_scene_obs = (curr_scene_obs - self.robot_state[:2]) * curr_scene_obs_mask
            state = np.concatenate((curr_scene_obs.flatten(), neight_matrix.flatten(), self.robot_state[2:], self.goal - self.robot_state[:2]))

        return state

    def get_dataset(self, agent='human', train = True, policy = None,
                    render = False, coll_done = False, action_norm_dataset = True, save_render = False):
        '''
        Main function to gather all data executed by the offline agent. This data can be evaluated or used for
        offline rl training.
        '''
        actions_rel, observations, rewards, \
        terminals, scenes, robot_ids, actions, goals_dis, goals = [], [], [], [], [], [], [], [], []
        success, path_len_h, path_len_r, time_out, inference_t = [], [], [], [], []
        for j, batch in enumerate(tqdm(self.loader)):
            state = self.init_scene(batch)
            # observations.append(state)
            collision = False
            robot_ids.append(self.robot_idx)
            scenes.append(self.hole_traj)
            goals_dis.append(np.linalg.norm(self.goal-self.start))
            goals.append(self.goal)
            action_traj = []
            last_info = ''
            for t in range(self.start_time+1, self.episode_time+1):
                if agent == 'human':
                    poposed_action = self.robot_traj_rel[t]
                    self.action = poposed_action
                elif agent =='robot':
                    if self.device == 'cuda':
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        poposed_action = policy.predict(np.expand_dims(state, axis=0))[0]
                        end.record()
                        torch.cuda.synchronize() # Waits for everything to finish running
                        #print(start.elapsed_time(end))
                        inference_t.append(start.elapsed_time(end))
                    else:
                        start_time = time.time()
                        poposed_action = policy.predict(np.expand_dims(state, axis=0))[0]
                        end_time = time.time()
                        duration = (end_time - start_time) * 1000.
                        #print(duration)
                        inference_t.append(duration)
                else:
                    print('Other´s are not implemented right now')
                observations.append(state)
                #next_state, reward, done, info = self.step(poposed_action, coll_done= coll_done, human = agent == 'human')
                if render:
                    if save_render:
                        self.save_render()
                    else:
                        self.render()
                        
                next_state, reward, done, _, info = self.step(poposed_action, coll_done= coll_done, human = agent == 'human')

                state = next_state
                if info == 'col':
                    collision = True
                action_traj.append(self.action)
                rewards.append(reward)
                terminals.append(done)
                if done:
                    last_info = info
                    break

            actions_rel.append(np.asarray(action_traj))
            assert last_info != ''  # goal or timeout, cannot be empty
            if agent == 'human':
                assert last_info != 'time_out'
            if not train: # gather all information for eval
                if not collision and last_info == 'goal':
                    success.append(1)
                else:
                    success.append(0)
                if not collision and last_info == 'time_out':
                    time_out.append(1)
                else:
                    time_out.append(0)
                rel_human_traj = self.robot_traj_rel[self.start_time+1: self.agent_hist + self.human_future]
                rel_robot_traj = np.array(action_traj)
                lengths_h = np.sqrt(np.sum(rel_human_traj ** 2, axis=1))  # Length between corners
                lengths_r = np.sqrt(np.sum(rel_robot_traj ** 2, axis=1))
                path_len_h.append(np.sum(lengths_h))
                path_len_r.append(np.sum(lengths_r))
                actions.append(self.start + np.cumsum(np.asarray(action_traj), axis=0))
        actions_rel = np.concatenate(actions_rel, axis=0 )
        rewards = np.asarray(rewards)
        observations = np.asarray(observations)
        terminals = np.asarray(terminals)
        robot_ids = np.asarray(robot_ids)


        if train:
            act = np.ascontiguousarray(actions_rel)
            if action_norm_dataset:
                act = self.min_max_norm(act)
            dataset = d3rlpy.dataset.MDPDataset(
                observations=np.ascontiguousarray(observations),
                actions=np.ascontiguousarray(act),
                rewards=np.ascontiguousarray(rewards),
                terminals=np.ascontiguousarray(terminals),
                transition_picker=CustomTransitionPicker()
            )
            return dataset
        else:
            return actions, scenes, robot_ids,\
                   np.array(goals_dis), np.array(goals), np.array(success),\
                   np.array(path_len_h), np.array(path_len_r), np.array(time_out), np.array(inference_t)