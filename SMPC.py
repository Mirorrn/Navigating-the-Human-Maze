import torch
from torch import nn
import numpy as np
import os

from model.NAR_with_Goal import TrajectoryGeneratorAR_goal
from model.NIAR_with_Goal import TrajectoryGeneratorGoalIAR
from model.NIAR import TrajectoryGenerator
from data.utils import batched_Robot_coll_smoothed_loss, cart2pol, pol2cart

class SMPC(nn.Module):
    def __init__(self):
        super(SMPC, self).__init__()

    def calc_cost_map_reward(self, trajectory, cost_map_obj):
        cost = cost_map_obj.get_cost_from_world_x_y(trajectory[:].cpu().numpy())
        cost = np.where(cost > 96, 100000, cost)

        return cost.sum(0).astype(int)

    def get_model(self):
        # Initialize and load pre-trained models
        _dir = os.path.dirname(__file__) or '.'
        _dir = _dir + "/model/weights/"

        if self.human_reaction:
            checkpoint_path = _dir + 'GCBC-univ_AR_Transformer_more_data/checkpoint_with_model.pt'
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
            self.model_g = TrajectoryGeneratorAR_goal(self.num_agent, self.robot_params_dict, self.dt,
                                                      predictions_steps=self.predictions_steps,
                                                      sample_batch=self.sample_batch,
                                                      device=self.device)
            self.model_g.load_state_dict(checkpoint["best_state"])
            self.model_g.to(self.device)
        else:
            checkpoint_path = _dir + 'GCBC-univ_NIAR_Transformer_more_data/checkpoint_with_model.pt'
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
            self.model_g = TrajectoryGeneratorGoalIAR(self.robot_params_dict, self.dt, self.device,
                                                      predictions_steps=self.predictions_steps,
                                                      sample_batch=self.sample_batch,
                                                      )
            self.model_g.load_state_dict(checkpoint["best_state"])
            self.model_g.to(self.device) 

        checkpoint_path = _dir + 'SIMNoGoal-univ_IAR_Full_trans/checkpoint_with_model.pt'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model_iar = TrajectoryGenerator(self.robot_params_dict, self.dt, self.device,
                                             predictions_steps=self.predictions_steps, sample_batch=self.sample_batch)
        self.model_iar.load_state_dict(checkpoint["best_state"])
        self.model_iar.to(self.device)
        self.model_iar.eval()

    def calc_r_goal_clamped(self, r_goal, last_obs_r_pos):
            goal_diff = (r_goal - last_obs_r_pos)
            v_g, yaw_g = cart2pol(goal_diff[:, 0], goal_diff[:, 1])
            v_g = torch.clamp(v_g, min=10, max=10.0)
            x_g, y_g = pol2cart(v_g.unsqueeze(-1), yaw_g.unsqueeze(-1))
            goal_clamped = torch.cat([x_g, y_g], dim=-1) + last_obs_r_pos

            return goal_clamped
    


    def calc_SI(self, pred_traj_abs_not_cond, pred_traj_abs_cond):
        pred_traj_abs_cond = pred_traj_abs_cond.reshape([self.predictions_steps, self.sample_batch, -1, 2])
        goal_reward = ((pred_traj_abs_not_cond.unsqueeze(1) - pred_traj_abs_cond) ** 2)   #.sum(dim=-1)
       # goal_cost = goal_cost.reshape(-1, self.sample_batch, self.numb_agents + 1)
        goal_reward = goal_reward[-1,1:,1:,:].sum(-1).sum(-1)

        return goal_reward
  
    def calc_goal_reward(self, goal, pred_traj_abs, robot_idx):
        goal_reward = ((goal - pred_traj_abs) ** 2).sum(dim=-1)
        goal_reward_robot = goal_reward[:,robot_idx].mean(0)
        return goal_reward_robot 

    def predict_human_goals(self, model, obs_traj_pos, traj_rel, robot_idx, r_pose, calc_new, z, sample_batch, costmap_obj=None):
        pred_traj_rel_for_goal, mu, _, _, _ = model(traj_rel, r_pose,
                                                            robotID=robot_idx,
                                                            z=z,
                                                            ar_step_or_DWA=False,
                                                            calc_new=calc_new,
                                                            mean_pred= False)
        pred_traj_abs = torch.cumsum(pred_traj_rel_for_goal, dim=0) + obs_traj_pos[-1]

        # calc next goal based on nll and costmap cost
        if costmap_obj:
            costmap_reward_human = torch.Tensor(self.calc_cost_map_reward(pred_traj_abs, costmap_obj))
        else:
            costmap_reward_human = 0.

        nll_h = ((pred_traj_rel_for_goal - mu) ** 2).sum(0).sum(-1)  # we do not use predicted scale, so each state has the same weigthening
        cost_h = (costmap_reward_human + nll_h).reshape(sample_batch, -1)
        ids_h = torch.argmin(cost_h, dim=0)
        tmp_num_agent = len(ids_h)
        seq_ids = ids_h * tmp_num_agent + torch.arange(tmp_num_agent, device=self.device)
        pred_traj_abs = pred_traj_abs[:, seq_ids].repeat(1, sample_batch, 1)

        return pred_traj_abs[-1]
        

    def calc_rewards(self, data, z, calc_new=True, costmap_obj=None, ar_step_or_DWA=True):
        # Calculate costs for different factors such as goal, collision, perturbation action and cost map
        with torch.no_grad():
            z = torch.cat([z[:,0].unsqueeze(1), z], dim = 1)
            obs_traj_pos, traj_rel, neigh_index, robot_idx, r_goal, r_pose = data[:6]
            if calc_new:
                self.goals = self.predict_human_goals(self.model_iar, obs_traj_pos, traj_rel, robot_idx, r_pose, calc_new, z, self.sample_batch)
                self.goals[robot_idx] = self.calc_r_goal_clamped(r_goal, obs_traj_pos[-1, robot_idx])

            # first entry in batch is for SI calculation, so robot is set to zero
            traj_rel[:,0] = 0.
            obs_traj_pos[:,0] = -20.

            pred_traj_rel, self.pertu_actions_clamped, nll = self.model_g(traj_rel, obs_traj_pos,
                                                                            neigh_index,
                                                                            r_pose, self.goals,
                                                                            robotID=robot_idx, z=z,
                                                                            optimize_latent_space = self.optimize_latent_space,
                                                                            mean_pred = True)

            pred_traj_abs = torch.cumsum(pred_traj_rel, dim=0) + obs_traj_pos[-1]
          
            pred_traj_abs_not_cond = pred_traj_abs[:, 0 : robot_idx[1]]


            nll = nll[1: ]

            goal_reward = self.calc_goal_reward(self.goals, pred_traj_abs, robot_idx)[1: ]
            SI_reward = self.calc_SI(pred_traj_abs_not_cond, pred_traj_abs)
        
            coll_reward = batched_Robot_coll_smoothed_loss(pred_traj_abs, self.sample_batch,
                                                         predictions_steps=self.predictions_steps,
                                                         batch=True,
                                                         collision_dist=self.robot_params_dict["collision_dist"]).view(
                                                         self.predictions_steps, -1).sum(0)
            if costmap_obj:
                costmap_reward = self.calc_cost_map_reward(pred_traj_abs[:, robot_idx], costmap_obj)
            else:
                costmap_reward = torch.zeros_like(goal_reward)


        return goal_reward, coll_reward[1: ], self.pertu_actions_clamped[:, 1: ], costmap_reward, pred_traj_rel[:,robot_idx], nll, SI_reward

    def get_pred_traj_abs(self):
        # Get the absolute predicted trajectories for agents
        # We need this for plotting in RVIZ on a real robot
        agent_future = self.pred_traj_abs.cpu().numpy().transpose(1, 0, 2)
        agent_future = agent_future.reshape(self.num_threads, self.sample_batch_per_thread, -1, self.predictions_steps, 2)
        agent_future = agent_future[
            self.min_thread_id, self._ids_for_plot[self.min_thread_id, :self.num_ids_for_plot]].mean(axis=0)
        return agent_future

    def predict(self,):
        return None

    def reset(self):
        self.U_init = torch.zeros([self.predictions_steps, self.sample_batch, 2], device=self.device)
