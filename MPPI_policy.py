import torch
import numpy as np
from data.social_nav_env import SocialNavEnv, evaluate
from SMPC import SMPC
from data.utils import prepare_states

def _ensure_non_zero(reward, beta, factor):
    return torch.exp(-factor * (reward - beta))


class Parallel_MPPI(SMPC):
    def __init__(self, robot_params_dict, human_reaction=True, dt=0.4, hist=8, number_samples = 800,
                 num_agent=5, coll_reward_gain=3000., SI_gain = 1000, goal_reward_gain = 0., nll_gain = 1.,
                 device='cuda'):
        super(Parallel_MPPI, self).__init__()
        self.robot_params_dict = robot_params_dict
        self.dt = dt
        self.device = device
        self.num_threads = 1
        self.sample_batch_per_thread = number_samples
        self.predictions_steps = 12
        self.max_iters = 1
        self.epsilon = 0.001
        self.alpha = 0.2
        self.num_elites = self.sample_batch_per_thread 
        self.lambda_ = 2.
        self.sample_batch = self.sample_batch_per_thread * self.num_threads + 1
        self.init_var = torch.ones([self.predictions_steps, self.sample_batch-1, 2], device=self.device)
        self.U_init = torch.zeros([self.predictions_steps, self.sample_batch-1, 2], device=self.device)
        self.human_reaction = human_reaction  # predict human reaction with ar model?
        self.index_serial = torch.arange(self.num_threads, device=self.device).unsqueeze(dim=-1).repeat(1,
                                                                                                        self.num_elites) * self.sample_batch_per_thread
    
        self.hist = hist
        self.num_agent = num_agent
        self.get_model()
        self.optimize_latent_space = False # not working well, do this with CEM!
        self.U_stop = torch.zeros(2, device=self.device)

        self.SI_gain = SI_gain
        self.coll_reward_gain = coll_reward_gain
        self.goal_reward_gain = goal_reward_gain
        self.nll_gain = nll_gain

    def predict(self, x, costmap_obj=None):
        # Perform prediction using the MPPI
        #with torch.cuda.amp.autocast():
        with torch.no_grad():

            self.costmap_obj = costmap_obj
            if self.costmap_obj is not None:
                self.real_robot = True
            else:
                self.real_robot = False

            x = torch.as_tensor(np.array(x), dtype=torch.float, device=self.device)
            x = x.repeat(self.sample_batch, 1)
            # The following line encodes the flattened input into tensors, e.g., observed states [num_observed_timesteps, batch_size, 2]
            x = prepare_states(x, self.hist, self.num_agent, device=self.device)
            var = self.init_var.clone()
            opt_count = 0
            calc_new = True
            U = self.U_init
            while (opt_count < self.max_iters):
                ar_step_or_DWA = opt_count % 2 == 0
                if torch.max(var) < self.epsilon and not ar_step_or_DWA:
                    break

                noise = var * torch.randn_like(var)
                pertu_actions = U + noise

                goal_reward, coll_reward, pertu_actions_clamped, costmap_reward, robot_traj, nll, SI_reward = self.calc_rewards(x,
                                                                                                                 pertu_actions.float(),
                                                                                                                 ar_step_or_DWA=ar_step_or_DWA,
                                                                                                                 calc_new=calc_new,
                                                                                                                 costmap_obj=self.costmap_obj)
                calc_new = False
                noise_clamped = pertu_actions_clamped - U
               # noise_clamped = noise_clamped[:, 1: ]
                reward = self.goal_reward_gain * goal_reward + self.coll_reward_gain * coll_reward + costmap_reward + self.nll_gain * nll + self.SI_gain*SI_reward

                reward_reshaped = reward.reshape(self.num_threads, self.sample_batch_per_thread)
                elite_reward, ids = torch.sort(reward_reshaped, descending=False)
                reward_reshaped = elite_reward[:, :self.num_elites]
                elite_ids = ids[:, :self.num_elites] + self.index_serial
                elite_ids = elite_ids.reshape(-1)
                elite_noise = noise_clamped[:, elite_ids].reshape(self.predictions_steps, self.num_threads,
                                                                  self.num_elites, 2)
                beta, batch_id = torch.min(reward_reshaped, dim=1)
                _, self.min_thread_id = torch.min(beta, dim=0)
                reward_total_non_zero = _ensure_non_zero(reward_reshaped, beta.unsqueeze(dim=-1), 1 / self.lambda_)
                eta = torch.sum(reward_total_non_zero, dim=1, keepdim=True)
                self.omega = ((1. / eta) * reward_total_non_zero).view(self.num_threads, self.num_elites, 1)
                perturbations = []
                for t in range(self.predictions_steps):
                    if self.optimize_latent_space:
                        perturbations.append(torch.sum(self.omega * robot_traj[t], dim=1))
                    else:
                        perturbations.append(torch.sum(self.omega * elite_noise[t], dim=1))
                perturbations = torch.stack(perturbations)
                perturbations = perturbations.unsqueeze(dim=2).repeat(1, 1, self.sample_batch_per_thread, 1).view(
                    self.predictions_steps, -1, 2)

                U = U + perturbations
                opt_count += 1

            if not self.optimize_latent_space:
                self.U_init = torch.roll(U, -1, 0)

            U = U[0, 0]
            if costmap_reward[elite_ids[0]] >= 100000 or coll_reward[elite_ids[0]] >= 100000:
                U = self.U_stop  # stop robot if the best predicted states are collisions

            if self.real_robot:
                return U
            else:
                return [U.cpu().numpy()]


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_env = SocialNavEnv(device=device, test_mode=True, use_robot_model=True, XYAction=False)
  
    policy = Parallel_MPPI(eval_env.robot_params_dict, dt=eval_env.dt, hist=eval_env.agent_hist, 
                           num_agent=eval_env.num_agents, device=device, 
                           human_reaction=True, coll_reward_gain=3000.,
                           SI_gain = 1000, goal_reward_gain = 0., nll_gain=100)

    evaluate(policy,  eval_env, eval_env.agent_hist, 
             eval_env.human_future, None, eval_env.goal_thresh, render=True, epoch=10, save_render=False) 