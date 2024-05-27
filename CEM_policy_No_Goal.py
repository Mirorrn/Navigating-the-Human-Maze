import torch
from torch import nn
import numpy as np
import os

from model.NAR import TrajectoryGeneratorAR
from model.NIAR import TrajectoryGenerator
from data.social_nav_env import SocialNavEnv, evaluate

torch.autograd.set_detect_anomaly(True)
from data.utils import prepare_states, batched_Robot_coll_smoothed_loss, cart2pol, pol2cart, actionXYtoROT,\
    batched_covariance, calc_cost_map_cost
from torch.distributions.multivariate_normal import MultivariateNormal


class CEM_IAR(nn.Module):
    def __init__(self, robot_params_dict, costmap_obj=None, dt=0.4, hist=7,
                 num_agent=5, obstacle_cost_gain=1000, mode='iar', mean_pred=False,
                 device='cuda', bc=False):
        super(CEM_IAR, self).__init__()
        self.robot_params_dict = robot_params_dict
        if self.robot_params_dict['use_robot_model']:
            print('Robot model is used. Compute linear and angular velocities as next action')
        else:
            print('No, robot model used. Compute X and Y positions as next action')

        if mode in ['iar', 'ar', 'mix']:
            self.mode = mode
        else:
            print('Only supported modes are: iar, ar, mix')
        self.dt = dt
        self.bc= bc
        self.mean_pred = mean_pred
        self.optimize_latent_space = True
        self.device = device
        self.costmap_obj = costmap_obj
        self.sample_batch = 800
        if self.costmap_obj is not None:
            self.real_robot = True
        else:
            self.real_robot = False

        self.predictions_steps = 12
        self.init_mean = torch.zeros([self.predictions_steps, 2], device=self.device)
        self.init_var = torch.stack([torch.eye(2, device=self.device)] * self.predictions_steps)
        self.max_iters = 3
        self.epsilon = 0.001
        self.alpha = 0.5
        self.num_elites = 20
        self.obstacle_cost_gain = obstacle_cost_gain
        self.soft_update = True
        self.hist = hist
        self.num_agent = num_agent
        self.device = device
        self.get_model()
        self.plot_list = []
        self.w_goal = torch.tensor([[10], [10], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]],
                                   device=self.device)



    def get_model(self):

        _dir = os.path.dirname(__file__) or '.'
        _dir = _dir + "/model/weights/"

        checkpoint_path = _dir + 'SIMNoGoal-univ_fast_AR2/checkpoint_with_model.pt'

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model_ar = TrajectoryGeneratorAR(self.num_agent, self.robot_params_dict, self.dt,
                                         predictions_steps=self.predictions_steps,
                                         sample_batch=self.sample_batch,
                                         device=self.device)
        self.model_ar.load_state_dict(checkpoint["best_state"])
        if self.device == 'cuda':
            self.model_ar.cuda()
        else:
            self.model_ar.cpu()
        self.model_ar.eval()
        checkpoint_path = _dir + 'SIMNoGoal-univ_IAR_Full_trans/checkpoint_with_model.pt'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model_iar = TrajectoryGenerator(self.robot_params_dict, self.dt, self.device,
                                        predictions_steps=self.predictions_steps, sample_batch=self.sample_batch)
        self.model_iar.load_state_dict(checkpoint["best_state"])
        if self.device == 'cuda':
            self.model_iar.cuda()
        else:
            self.model_iar.cpu()
        self.model_iar.eval()

        return

    def calc_cost(self, data, z, goal=None, ar_step_or_DWA=True, calc_new=True, costmap_obj=None):

        with torch.no_grad():
            obs_traj_pos, traj_rel, neigh_index, robot_idx, r_goal, r_pose = data[:6]

            goal[robot_idx] = r_goal
            if self.mode == 'iar':
                pred_traj_rel, self.mu, scale, clamped_action, nll = self.model_iar(traj_rel, r_pose, robotID=robot_idx,
                                                                                    z=z,
                                                                                    ar_step_or_DWA=ar_step_or_DWA,
                                                                                    calc_new=calc_new,
                                                                                    optimize_latent_space=self.optimize_latent_space,
                                                                                    mean_pred = False,)
            elif self.mode == 'mix':
                pred_traj_rel, mu, scale, clamped_action, nll = self.model_iar(traj_rel, r_pose, robotID=robot_idx,
                                                                               z=z,
                                                                               ar_step_or_DWA=False,
                                                                               calc_new=calc_new,
                                                                               optimize_latent_space=self.optimize_latent_space,
                                                                               mean_pred=False, )


                if ar_step_or_DWA:
                    proposed_robot_action = pred_traj_rel[:, robot_idx]
                    pred_traj_rel, clamped_action, nll = self.model_ar(traj_rel, obs_traj_pos,
                                                                                            neigh_index,
                                                                                            r_pose,
                                                                                            proposed_robot_action=proposed_robot_action,
                                                                                            robotID=robot_idx, z=z,
                                                                                            optimize_latent_space=self.optimize_latent_space,
                                                                                            mean_pred = True,)

            elif self.mode == 'ar':
                pred_traj_rel, clamped_action, nll = self.model_ar(traj_rel, obs_traj_pos,
                                                                                        neigh_index,
                                                                                        r_pose,
                                                                                        robotID=robot_idx, z=z,
                                                                                        optimize_latent_space=self.optimize_latent_space,
                                                                                        mean_pred = True,)

         #   test = (pred_traj_rel[:, robot_idx] - pred_traj_rel2[:, robot_idx]).sum()
            self.pred_traj_abs = torch.cumsum(pred_traj_rel, dim=0) + obs_traj_pos[-1]

            goal_diff = (goal[robot_idx] - obs_traj_pos[-1, robot_idx])
            v_g, yaw_g = cart2pol(goal_diff[:, 0], goal_diff[:, 1])
            v_g = torch.clamp(v_g, max=3.0)
            x_g, y_g = pol2cart(v_g, yaw_g)
            goal_clamped = torch.cat([x_g.unsqueeze(dim=1), y_g.unsqueeze(dim=1)], dim=-1) + obs_traj_pos[-1, robot_idx]




            goal_cost = ((goal_clamped - self.pred_traj_abs[:, robot_idx]) ** 2).sum(dim=-1)
            goal_cost = (goal_cost * self.w_goal).mean(0)
           # goal_cost = (goal_cost).mean(0)
            coll_cost = batched_Robot_coll_smoothed_loss(self.pred_traj_abs, self.sample_batch,
                                                         predictions_steps=self.predictions_steps,
                                                         batch=True,
                                                         collision_dist=self.robot_params_dict["collision_dist"]).view(
                self.predictions_steps, -1).sum(0)

            if costmap_obj:
                costmap_cost = calc_cost_map_cost(self.pred_traj_abs[:, robot_idx], costmap_obj, 0)
            else:
                costmap_cost = 0.

        #self.plot_list.append([obs_traj_pos, self.pred_traj_abs, goal_clamped, robot_idx])
        return goal_cost, coll_cost, costmap_cost, nll, pred_traj_rel[:, robot_idx], clamped_action

    def get_pred_traj_abs(self):
        agent_future = self.pred_traj_abs.cpu().numpy().transpose(1, 0, 2)
        agent_future = agent_future.reshape(self.sample_batch, -1, self.predictions_steps, 2)
        agent_future = agent_future[self.best_id[0]]#.mean(axis=0)
        return agent_future

    def predict(self, x):
        with torch.no_grad():
            x = torch.as_tensor(np.array(x), dtype=torch.float, device=self.device)
            x = x.repeat(self.sample_batch, 1)
            # if CEM is used for bc, prep_state will produce an additional next state which we do not need here
            x = prepare_states(x, self.hist, self.num_agent, bc=self.bc, device=self.device)
            pred_traj_fake_goal = torch.zeros_like(x[0][0])
            mean = self.init_mean.clone()
            var = self.init_var.clone()
            opt_count = 0
            calc_new = True
            while (opt_count < self.max_iters):
                self.action_distribution = MultivariateNormal(mean, covariance_matrix=var)
                samples = self.action_distribution.sample((self.sample_batch,)).permute(1, 0, 2)
                ar_step_or_DWA = opt_count % 2 == 0
                goal_cost, coll_cost, costmap_cost, nll, pred_traj_rel, clamped_action = self.calc_cost(x, samples,
                                                                                                          pred_traj_fake_goal,
                                                                                                          ar_step_or_DWA=ar_step_or_DWA,
                                                                                                          calc_new=calc_new,
                                                                                                          costmap_obj=self.costmap_obj)
                calc_new = False
                cost = 1 * goal_cost + self.obstacle_cost_gain * coll_cost + 1 * costmap_cost + nll
                best_ids = torch.argsort(cost, descending=False)
                elites = samples[:, best_ids][:, :self.num_elites]
                new_mean = torch.mean(elites, dim=1)
                new_var = batched_covariance(elites.permute(0, 2, 1))
                if self.soft_update:
                    # soft update
                    mean = (self.alpha * mean + (1. - self.alpha) * new_mean)#.unsqueeze(dim=1).repeat(1,
                                                                                                         #   self.sample_batch,
                                                                                                         #   1)
                    var = (self.alpha * var + (1. - self.alpha) * new_var)#.unsqueeze(dim=1).repeat(1,
                                                                                                       #  self.sample_batch,
                                                                                                       #  1)
                else:
                    mean = new_mean
                    var = new_var
                opt_count += 1

            self.best_id = best_ids[:self.num_elites]
            if self.real_robot:
                if costmap_cost[self.best_id[0]] >= 100000 or coll_cost[self.best_id[0]] >= 100000 :
                    #print('Stop!')
                    return np.array([0., 0.])  # stop robot if more than 50% of predicted states are collisions


            best_action = pred_traj_rel[:, self.best_id].mean(dim=1)
            r_pose = x[5]

            if self.robot_params_dict['use_robot_model']:
                U = actionXYtoROT(best_action[0].unsqueeze(dim=0), r_pose[0].unsqueeze(dim=0),
                              self.dt).squeeze().cpu().numpy()
            else:
                U = best_action[0].cpu().numpy()

            if self.real_robot:
                return U
            else:
                return [U]

    def reset(self):
        return None


if __name__ == '__main__':
    # env = SocialNavEnv(bc=True)
    device = 'cpu'
    eval_env = SocialNavEnv(device=device, test_mode=True, use_robot_model=True, scene_mode='long', XYAction=False)
    policy = CEM_IAR(eval_env.robot_params_dict, dt=eval_env.dt, hist=eval_env.agent_hist,
                     num_agent=eval_env.num_agents, device=device, mode='mix')

    evaluate(policy, eval_env, eval_env.agent_hist,
             eval_env.human_future, eval_env.goal_thresh, render=False, epoch=10)