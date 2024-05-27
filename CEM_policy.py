import torch
from torch import nn
import numpy as np
import os

from model.NAR_with_Goal import TrajectoryGeneratorAR_goal
from model.NIAR import TrajectoryGenerator
from model.NIAR_with_Goal import TrajectoryGeneratorGoalIAR
from data.social_nav_env import SocialNavEnv, evaluate

torch.autograd.set_detect_anomaly(True)
from data.utils import prepare_states, batched_Robot_coll_smoothed_loss, actionXYtoROT,\
    batched_covariance
from torch.distributions.multivariate_normal import MultivariateNormal
from SMPC import SMPC

class CEM_Policy(SMPC):
    def __init__(self, robot_params_dict, costmap_obj=None, dt=0.4, hist=7, number_samples = 800,
                 num_agent=5, coll_reward_gain=3000., goal_all_w = 1., goal_reward_gain = 1., nll_gain = 1.,
                 mode='iar', mean_pred=False, optimize_latent_space = False,
                 device='cuda', bc=False):
        super(CEM_Policy, self).__init__()
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
        self.optimize_latent_space = optimize_latent_space
        self.device = device
        self.costmap_obj = costmap_obj
        self.sample_batch = number_samples
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
        self.num_elites = 5   # do not go below 3 elites, cause -> because covariance computation will not work 
        self.soft_update = False
        self.hist = hist
        self.num_agent = num_agent
        self.device = device
        self.get_model()

        self.goal_all_w = goal_all_w
        self.coll_reward_gain = coll_reward_gain
        self.goal_reward_gain = goal_reward_gain
        self.nll_gain = nll_gain

    # override
    def get_model(self):

        _dir = os.path.dirname(__file__) or '.'
        _dir = _dir + "/model/weights/"

        checkpoint_path = _dir + 'GCBC-univ_AR_Transformer_more_data/checkpoint_with_model.pt'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model_ar_g = TrajectoryGeneratorAR_goal(self.num_agent, self.robot_params_dict, self.dt,
                                         predictions_steps=self.predictions_steps,
                                         sample_batch=self.sample_batch,
                                         device=self.device)
        self.model_ar_g.load_state_dict(checkpoint["best_state"])
        if self.device == 'cuda':
            self.model_ar_g.cuda()
        else:
            self.model_ar_g.cpu()
        self.model_ar_g.eval()


        checkpoint_path = _dir + 'GCBC-univ_NIAR_Transformer_more_data/checkpoint_with_model.pt'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model_iar_g = TrajectoryGeneratorGoalIAR(self.robot_params_dict, self.dt, self.device,
                                                      predictions_steps=self.predictions_steps,
                                                      sample_batch=self.sample_batch,
                                                      )
        self.model_iar_g.load_state_dict(checkpoint["best_state"])
        if self.device == 'cuda':
            self.model_iar_g.cuda()
        else:
            self.model_iar_g.cpu()
        self.model_iar_g.eval()

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

    # override
    def calc_rewards(self, data, z, ar_step_or_DWA=True, calc_new=True, costmap_obj=None):

        with torch.no_grad():
            obs_traj_pos, traj_rel, neigh_index, robot_idx, r_goal, r_pose = data[:6]

            
            # calc robot goal first
            if calc_new:
                goals = self.predict_human_goals(self.model_iar, obs_traj_pos, traj_rel, robot_idx, r_pose, calc_new, z, self.sample_batch)
                goals[robot_idx] = self.calc_r_goal_clamped(r_goal, obs_traj_pos[-1, robot_idx])
                self.goals = goals





            if self.mode == 'iar':
                pred_traj_rel, clamped_action, nll = self.model_iar_g(traj_rel, obs_traj_pos,
                                                                                    neigh_index, r_pose, self.goals,
                                                                                    robotID=robot_idx, z=z,
                                                                                   # ar_step_or_DWA=ar_step_or_DWA,
                                                                                    calc_new=calc_new,
                                                                                    optimize_latent_space=self.optimize_latent_space,
                                                                                    mean_pred = True,)
                
            elif self.mode == 'mix':
                pred_traj_rel, clamped_action, nll = self.model_iar_g(traj_rel, obs_traj_pos,
                                                                                    neigh_index, r_pose, self.goals,
                                                                                    robotID=robot_idx, z=z,
                                                                                    ar_step_or_DWA= not ar_step_or_DWA,
                                                                                    calc_new=calc_new,
                                                                                    optimize_latent_space=self.optimize_latent_space,
                                                                                    mean_pred = False,)


                if ar_step_or_DWA:
                    proposed_robot_action = pred_traj_rel[:, robot_idx]
                    pred_traj_rel, clamped_action, nll = self.model_ar_g(traj_rel, obs_traj_pos,
                                                                                    neigh_index,
                                                                                    r_pose, self.goals,
                                                                                    proposed_robot_action=proposed_robot_action,
                                                                                    robotID=robot_idx, z=z,
                                                                                    optimize_latent_space=self.optimize_latent_space,
                                                                                    mean_pred = False,)

            elif self.mode == 'ar':
                pred_traj_rel, clamped_action, nll = self.model_ar_g(traj_rel, obs_traj_pos,
                                                                                        neigh_index,
                                                                                        r_pose, self.goals,
                                                                                        robotID=robot_idx, z=z,
                                                                                        optimize_latent_space=self.optimize_latent_space,
                                                                                        mean_pred = True,)

            pred_traj_abs = torch.cumsum(pred_traj_rel, dim=0) + obs_traj_pos[-1]
            self.pred_traj_abs = pred_traj_abs # for plotting in rviz
            goal_reward = self.calc_goal_reward(self.goals, pred_traj_abs, robot_idx)


            coll_reward = batched_Robot_coll_smoothed_loss(pred_traj_abs, self.sample_batch,
                                                         predictions_steps=self.predictions_steps,
                                                         batch=True,
                                                         collision_dist=self.robot_params_dict["collision_dist"]).view(
                                                         self.predictions_steps, -1).sum(0)
            if costmap_obj:
                costmap_reward = self.calc_cost_map_reward(pred_traj_abs[:, robot_idx], costmap_obj)
            else:
                costmap_reward = torch.zeros_like(goal_reward)

        #self.plot_list.append([obs_traj_pos, self.pred_traj_abs, goal_clamped, robot_idx])
        return goal_reward, coll_reward, costmap_reward, nll, pred_traj_rel[:, robot_idx], clamped_action

    # override
    def predict(self, x):
        with torch.no_grad():
            x = torch.as_tensor(np.array(x), dtype=torch.float, device=self.device)
            x = x.repeat(self.sample_batch, 1)
            # if CEM is used for bc, prep_state will produce an additional next state which we do not need here
            x = prepare_states(x, self.hist, self.num_agent, device=self.device)
            mean = self.init_mean.clone()
            var = self.init_var.clone()
            opt_count = 0
            calc_new = True
            while (opt_count < self.max_iters):
                self.action_distribution = MultivariateNormal(mean, covariance_matrix=var)
                samples = self.action_distribution.sample((self.sample_batch,)).permute(1, 0, 2)
                ar_step_or_DWA = opt_count % 2 == 0
                goal_reward, coll_reward, costmap_reward, nll, pred_traj_rel, _ = self.calc_rewards(x, samples,
                                                                                                ar_step_or_DWA=ar_step_or_DWA,
                                                                                                calc_new=calc_new,
                                                                                                costmap_obj=self.costmap_obj)
                calc_new = False
                reward = self.goal_reward_gain * goal_reward + self.coll_reward_gain * coll_reward + costmap_reward + self.nll_gain * nll

                best_ids = torch.argsort(reward, descending=False)
                elites = samples[:, best_ids][:, :self.num_elites]
                new_mean = torch.mean(elites, dim=1)
                new_var = batched_covariance(elites.permute(0, 2, 1))
                if self.soft_update:
                    # soft update
                    mean = (self.alpha * mean + (1. - self.alpha) * new_mean)
                    var = (self.alpha * var + (1. - self.alpha) * new_var)
                else:
                    mean = new_mean
                    var = new_var
                opt_count += 1

            self.best_id = best_ids[:self.num_elites]
            if self.real_robot:
                if costmap_reward[self.best_id[0]] >= 100000 or coll_reward[self.best_id[0]] >= 100000 :
                    return np.array([0., 0.]) 


            best_action = pred_traj_rel[:, self.best_id].mean(dim=1)

            if not self.optimize_latent_space:
                self.init_mean = torch.roll(best_action, -1, 0)


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



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_env = SocialNavEnv(device=device, test_mode=True, use_robot_model=True, XYAction=False)
    policy = CEM_Policy(eval_env.robot_params_dict, dt=eval_env.dt, hist=eval_env.agent_hist, optimize_latent_space = False,
                     num_agent=eval_env.num_agents, device=device, mode='mix', coll_reward_gain=3000., goal_reward_gain = 0., nll_gain=100., )

    evaluate(policy,  eval_env, eval_env.agent_hist, 
             eval_env.human_future, None, eval_env.goal_thresh, render=False, epoch=10)