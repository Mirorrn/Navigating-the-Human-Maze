import torch
import numpy as np
import d3rlpy
import torch, math
import os
from typing_extensions import Protocol
from typing import Any, List, Tuple, Union, cast



#**********************************************************************************************

# Example TransitionPicker that simply picks transition
class CustomTransitionPicker(d3rlpy.dataset.TransitionPickerProtocol):
    def __call__(self, episode: d3rlpy.dataset.EpisodeBase, index: int) -> d3rlpy.dataset.Transition:
        observation = episode.observations[index]
        is_terminal = episode.terminated and index == episode.size() - 1
        if is_terminal:
            next_observation = episode.observations[index]
        else:
            next_observation = episode.observations[index + 1]


        
        # compute return-to-go
        length = episode.size() - index
        cum_gammas = np.expand_dims(0.95** np.arange(length), axis=1)
        return_to_go = np.sum(cum_gammas * episode.rewards[index:], axis=0)

        return d3rlpy.dataset.Transition(
            observation=observation,
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=next_observation,
            return_to_go=return_to_go,
            terminal=float(is_terminal),
            interval=1,
        )


#**********************************************************************************************

def GaußNLL(mu, scale, pred_traj_gt):
    var = torch.exp(scale).pow(2)

    loss = 0.5 * math.log(2 * math.pi) + 0.5 * (torch.log(var) + (pred_traj_gt - mu) ** 2 / var)

    return loss#.mean()

def batched_Robot_coll_smoothed_loss(pred_batch, sample_batch, predictions_steps =12, collision_dist=0.2, batch=False):
    if batch:
        currSzene = pred_batch.contiguous().reshape(sample_batch * predictions_steps, -1, 2)
    else:
        currSzene = pred_batch.contiguous().reshape(sample_batch, -1, 2)
    dist_mat = torch.sqrt(((currSzene[:, 0].unsqueeze(dim=1) - currSzene[:, 1:] ) ** 2).sum(dim=-1))#.sum(dim=0)
    dist_mat = 1. - torch.sigmoid((dist_mat - collision_dist) * 35.)
    dist_mat = torch.where(dist_mat > 0.8, torch.tensor(100000., device=dist_mat.device), dist_mat)
    # dist_mat = torch.logical_and(0. != dist_mat, dist_mat < collision_dist)
    dist_mat = dist_mat.sum(dim=-1, keepdim=True)  # get number of coll for every pedestrian traj

    return dist_mat

def actionXYtoROT(actionXY, robot_state, dt):
    # robot_state state[v_x(m), v_y(m), yaw(rad), v(m / s), omega(rad / s)] x and y are displacments to last state
    v, yaw = cart2pol(actionXY[:, 0], actionXY[:, 1])
    yaw_r = robot_state[:, 2]
    diff = torch.atan2(torch.sin(yaw-yaw_r), torch.cos(yaw-yaw_r))

    omega_t = (diff) / dt
    v_t = v / dt
    return torch.cat([v_t.unsqueeze(dim=-1), omega_t.unsqueeze(dim=-1)], dim=-1)

def prepare_states(x, hist, num_agent, full_return= False, device='cuda'):
    # deserialize the input sequence. Care empty sequences are here filtered out for performance increase
    with torch.no_grad():
        zero_start = torch.zeros(1, device=device, dtype=torch.int)
        batch = x.shape[0]
        n = (2 * (num_agent + 1) * (hist + 1))
        robot_state = x[:, -7:-2]
        r_goal = x[:, -2:]

        neigh_matrix_vec = x[:, n:-7]
        neigh_matrix = neigh_matrix_vec.reshape(batch, num_agent + 1, num_agent +1)

        obs_traj_pos = x[:, :n].reshape(batch, hist +1 , num_agent +1 , 2)
        inx_peds_in_seq = neigh_matrix[:,0,:]
        inx_peds_in_seq[:,0] = True
        inx_peds_in_seq = inx_peds_in_seq.bool()
        obs_traj_pos = obs_traj_pos.permute(1, 0, 2, 3).reshape(hist+1, -1, 2)
        cum_start_idx = torch.cumsum(torch.cat([zero_start, inx_peds_in_seq.sum(dim=-1)]),
                                        dim=-1)

        inx_peds_in_seq = inx_peds_in_seq.reshape(-1)
        obs_traj_pos = obs_traj_pos[:,inx_peds_in_seq]
        mask_rel = torch.where(obs_traj_pos != 0, True, False)
        traj_rel = torch.zeros_like(obs_traj_pos)
        mask_rel_first_element = mask_rel.logical_not() * 999.99
        obs_traj_pos_filled = obs_traj_pos + mask_rel_first_element
        traj_rel[1:] = (obs_traj_pos_filled[1:] - obs_traj_pos_filled[:-1])
        traj_rel = torch.where(traj_rel < -300, torch.zeros_like(obs_traj_pos), traj_rel) * mask_rel

        neigh_indx = torch.argwhere(neigh_matrix != 0)
        neigh_indx_shifted = cum_start_idx[neigh_indx[:, 0]]
        neigh_indx[:, 0:2] = neigh_indx[:, 1:] + neigh_indx_shifted.unsqueeze(dim=1)


        data = [obs_traj_pos, traj_rel, neigh_indx, cum_start_idx[:-1], r_goal, robot_state]
    if full_return:
        data.append(inx_peds_in_seq)
    return data



def cart2pol(x, y):
    rho = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.arctan2(y, x) 
    return rho, phi

def pol2cart(rho, phi):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y

def dynamic_window(robot_state, u, params_dict, dt):
    # robot_state state[v_x(m / s), v_y(m / s), yaw(rad), v(m / s), omega(rad / s)] x and y are displacments to last state
    # Dynamic window from robot specification
    batch = u.shape[0]
    theta_current = robot_state[:, 2].unsqueeze(1)
    v_t = robot_state[:, 3].unsqueeze(1)
    omega = robot_state[:, 4].unsqueeze(1)
    v_new = u[:, 0].unsqueeze(1)
    yaw_new = u[:, 1].unsqueeze(1)
    filler = torch.ones_like(robot_state[:, 4]).unsqueeze(1)

    Vs = [params_dict["min_speed"] * filler, params_dict["max_speed"] * filler,
          -params_dict["max_yaw_rate"] * filler, params_dict["max_yaw_rate"] * filler]
    # Dynamic window from motion model
    Vd = [v_t - params_dict["max_accel"] * dt,
          v_t + params_dict["max_accel"] * dt,
          omega - filler * params_dict["max_delta_yaw_rate"] * dt,
          omega + filler * params_dict["max_delta_yaw_rate"] * dt]

    v_min = torch.max(torch.cat([Vs[0], Vd[0]], dim=1), dim=1)[0].unsqueeze(dim=1)
    v_max = torch.min(torch.cat([Vs[1], Vd[1]], dim=1), dim=1)[0].unsqueeze(dim=1)
    yaw_rate_min = torch.max(torch.cat([Vs[2], Vd[2]], dim=1), dim=1)[0].unsqueeze(dim=1)
    yaw_rate_max = torch.min(torch.cat([Vs[3], Vd[3]], dim=1), dim=1)[0].unsqueeze(dim=1)
    dw = [v_min, v_max, yaw_rate_min, yaw_rate_max]

    v_new = torch.clamp(v_new, min=dw[0], max=dw[1])
    yaw_new = torch.clamp(yaw_new, min=dw[2], max=dw[3])
    theta = (yaw_new*dt + theta_current)
    v_x, v_y = pol2cart(v_new, theta)
    new_robot_state = torch.cat([v_x, v_y, theta, v_new, yaw_new], dim=-1)
    return new_robot_state


def get_dset_path(dset_type, dset_gupta):
    if dset_gupta:
        _dir = os.path.dirname(__file__)
        return os.path.join(_dir, 'univ', dset_type)
    else:
        _dir = os.path.dirname(__file__)
        return os.path.join(_dir, 'univ-big', dset_type)

def fast_coll_counter(scene, robot_id, coll_dist= 0.2): # todo: check only robot coll!!!!!
    dist_mat = torch.cdist(scene, scene, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
    dist_mat = dist_mat[:, robot_id]
    filter_zeros = torch.logical_and(0. != dist_mat,  dist_mat < coll_dist)
    return (filter_zeros.sum(), dist_mat)


def robot_dist_for_Eval(scene, robot_id): 
    dist_mat = torch.cdist(scene, scene, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
    dist_mat = dist_mat[:, robot_id][:,0].numpy()
    dist_mat = np.where(dist_mat == 0., 99999999, dist_mat)
    # dist_mat = dist_mat[0. != dist_mat].numpy()
    density = np.where(dist_mat < 3., True, False)
    density = density.sum(axis=-1) # .mean()
    min_distance = np.min(dist_mat)
    dist_less21 = min_distance < 0.2
    dist_less3 = min_distance < 0.3
    return min_distance, dist_less21.sum(), dist_less3.sum(),density

class AlgoProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def gamma(self) -> float:
        ...

class eval_helper():
    def __init__(self, test_after_n_epochs=5):
        self.test_after_n_epochs = test_after_n_epochs
        self.Epoch_counter = 1
        self.new_sucess_high = 0.
    def evaluate_test(self, env, agent_hist=2, human_future=12, goal_thresh=0.2):
        def scorer(algo: AlgoProtocol, *args: Any) -> float:
            if self.Epoch_counter % self.test_after_n_epochs != 0:
                self.Epoch_counter += 1
                return 0
            self.Epoch_counter += 1
            actions, scenes, robot_ids, \
            goals_dis, goals, success, path_len_h, \
            path_len_r, timeout, inference_t = env.get_dataset(agent='robot', policy=algo, train=False)
            min_distance_s_r, less_2_s_r, less_3_s_r, densities_r = [], [], [], []
            min_distance_s_h, less_2_s_h, less_3_s_h, densities_h = [], [], [], []
            # path_len_h, path_len_r, sucess = [], [], []
            for i, scene in enumerate(scenes):
                scene_with_robot = scene.copy()
                scene_with_robot[agent_hist + 1:agent_hist + actions[i].shape[0] + 1, robot_ids[i][0]] = actions[i]
                scene_with_robot = torch.from_numpy(scene_with_robot)
                min_distance_r, dist_less21_r, dist_less3_r, density_r = robot_dist_for_Eval(
                    scene_with_robot[agent_hist + 1:agent_hist + actions[i].shape[0] + 1], robot_ids[i])
                assert success[i] == 1 - (timeout[i] + (dist_less21_r.sum() > 0))

                min_h, dist_less21_h, dist_less3_h, density_h = robot_dist_for_Eval(
                    torch.from_numpy(scene[agent_hist + 1:agent_hist + human_future]), robot_ids[i])
                min_distance_s_r.append(min_distance_r)
                less_2_s_r.append(dist_less21_r)
                less_3_s_r.append(dist_less3_r)
                densities_r.append(density_r)

                min_distance_s_h.append(min_h)
                less_2_s_h.append(dist_less21_h)
                less_3_s_h.append(dist_less3_h)
                densities_h.append(density_h)
            # distance Robot
            mean_sucess_r = np.mean(np.array(success))
            mean_timeout = np.mean(timeout)
            min_distance_s_r = np.asarray(min_distance_s_r)
            path_len_r = np.asarray(path_len_r)
            path_len_r_mean = np.mean(path_len_r)
            # less_2_s_r = np.asarray(less_2_s_r).sum()
            less_3_s_r = np.asarray(less_3_s_r).sum()
            mean_min_distance_s_r = np.mean(min_distance_s_r)
            perc_less_21_r = np.mean(np.array(less_2_s_r))  # / len(min_s_r)
            test = mean_sucess_r + perc_less_21_r + mean_timeout
            perc_less_3_r = less_3_s_r / len(min_distance_s_r)
            densities_r = np.concatenate(densities_r).mean()
            # distance human
            min_distance_s_h = np.asarray(min_distance_s_h)
            path_len_h = np.asarray(path_len_h)
            path_len_h_mean = np.mean(path_len_h)
            less_2_s_h = np.asarray(less_2_s_h).sum()
            less_3_s_h = np.asarray(less_3_s_h).sum()
            mean_min_s_h = np.mean(min_distance_s_h)
            perc_less_21_h = less_2_s_h / len(min_distance_s_h)
            perc_less_3_h = less_3_s_h / len(min_distance_s_h)
            densities_h = np.asarray(densities_h).mean()
            path_q = path_len_r / (path_len_h + 1e-12)
            longer_as_humans = (path_q > 1.25).sum() / len(min_distance_s_h)  # ToDo some path´s are zero long....
            max_path_longer_as_human = np.max(path_q)
            goals_mean = goals_dis.mean()
            goals_min = goals_dis.min()
            goals_max = goals_dis.max()
            # return actions based on the greedy-policy
            # action = dqn.predict([observation])[0]
            if mean_sucess_r > self.new_sucess_high:
                print('---------------NEW HIGHSCORE-------------------')
                self.new_sucess_high = mean_sucess_r
            results = {
                "Sucess_highscore": self.new_sucess_high,
                "mean_sucess_r": mean_sucess_r,
                "mean_timeout": mean_timeout,
                "mean_min_distance_s_r": mean_min_distance_s_r,
                "perc_less_21_r": perc_less_21_r,
                "perc_less_3_r": perc_less_3_r,
                # "path_len_r_mean": path_len_r_mean,
                # "densities_r": densities_r,
                # "mean_min_s_h": mean_min_s_h,
                # "perc_less_21_h": perc_less_21_h,
                # "perc_less_3_h": perc_less_3_h,
                # "path_len_h_mean": path_len_h_mean,
                # "densities_h": densities_h,
                # "longer_as_humans_path": longer_as_humans,
                # "max_path_longer_as_human": max_path_longer_as_human,
                # "goals_mean": goals_mean,
                # "goals_min": goals_min,
                # "goals_max": goals_max,
            }
            print("{:<4} {:<4}".format('Label', 'Number'))
            for k, v in results.items():
                # label, num = v
                print("{:<4}  {:.4f}".format(k, v))
            for k, v in results.items():
                # label, num = v
                print("{:.4f}".format(v))
            return mean_sucess_r

        return scorer

def evaluate(policy, env, agent_hist, human_future, goal_thresh= 0.2, verbose = True, render=False, epoch=1, save_render=False):
    epoch_results = []
    for e in range(epoch):
        print('Start epoch ' +str(e + 1) + '/' + str(epoch))
        actions, scenes, robot_ids, \
        goals_dis, goals, success, path_len_h, \
        path_len_r, timeout, inference_t = env.get_dataset(agent='robot', policy=policy, train=False, render=render, save_render=save_render)
        min_distance_s_r, less_2_s_r, less_3_s_r, densities_r = [], [], [], []
        min_distance_s_h, less_2_s_h, less_3_s_h, densities_h = [], [], [], []
        for i, scene in enumerate(scenes):
            scene_with_robot = scene.copy()
            scene_with_robot[agent_hist+1:agent_hist + actions[i].shape[0]+1, robot_ids[i][0]] = actions[i]
            scene_with_robot = torch.from_numpy(scene_with_robot)
            min_r, dist_less21_r, dist_less3_r, density_r = robot_dist_for_Eval(
                scene_with_robot[agent_hist + 1:agent_hist + actions[i].shape[0]+1], robot_ids[i])
            assert success[i] == 1- (timeout[i] + (dist_less21_r.sum() > 0))
            min_h, dist_less21_h, dist_less3_h, density_h = robot_dist_for_Eval(
                torch.from_numpy(scene[agent_hist + 1:agent_hist + human_future]), robot_ids[i])
            min_distance_s_r.append(min_r)
            less_2_s_r.append(dist_less21_r)
            less_3_s_r.append(dist_less3_r)
            densities_r.append(density_r)

            min_distance_s_h.append(min_h)
            less_2_s_h.append(dist_less21_h)
            less_3_s_h.append(dist_less3_h)
            densities_h.append(density_h)
        # distance Robot
        mean_sucess_r = np.mean(np.array(success))
        mean_timeout = np.mean(timeout)
        min_distance_s_r = np.asarray(min_distance_s_r)
        path_len_r = np.asarray(path_len_r)
        path_len_r_mean = np.mean(path_len_r)
        inference_t_mean = np.mean(inference_t)
        # less_2_s_r = np.asarray(less_2_s_r).sum()
        less_3_s_r = np.asarray(less_3_s_r).sum()
        mean_distance_min_s_r = np.mean(min_distance_s_r)
        perc_less_21_r = np.mean(np.array(less_2_s_r)) # / len(min_s_r)
        perc_less_3_r = less_3_s_r / len(min_distance_s_r)
        densities_r = np.concatenate(densities_r).mean()
        # distance human
        min_distance_s_h = np.asarray(min_distance_s_h)
        path_len_h = np.asarray(path_len_h)
        path_len_h_mean = np.mean(path_len_h)
        less_2_s_h = np.asarray(less_2_s_h).sum()
        less_3_s_h = np.asarray(less_3_s_h).sum()
        mean_distance_min_s_h = np.mean(min_distance_s_h)
        perc_less_21_h = less_2_s_h / len(min_distance_s_h)
        perc_less_3_h = less_3_s_h / len(min_distance_s_h)
        densities_h = np.asarray(densities_h).mean()
        path_q = path_len_r / (path_len_h + 1e-12)
        longer_as_humans = (path_q > 1.25).sum() / len(min_distance_s_h) # ToDo some path´s are zero long....
        max_path_longer_as_human = np.max(path_q)
        goals_mean = goals_dis.mean()
        goals_min = goals_dis.min()
        goals_max = goals_dis.max()
        if verbose:
            results = {
                "mean_sucess_r": mean_sucess_r,
                "mean_timeout": mean_timeout,
                "mean_distance_min_s_r": mean_distance_min_s_r,
                "perc_less_21_r": perc_less_21_r,
                "perc_less_3_r": perc_less_3_r,
                "path_len_r_mean": path_len_r_mean,
                "densities_r": densities_r,
                "mean_distance_min_s_h": mean_distance_min_s_h,
                "perc_less_21_h": perc_less_21_h,
                "perc_less_3_h": perc_less_3_h,
                "path_len_h_mean": path_len_h_mean,
                "densities_h": densities_h,
                "longer_as_humans_path": longer_as_humans,
                "max_path_longer_as_human": max_path_longer_as_human,
                "goals_mean": goals_mean,
                "goals_min": goals_min,
                "goals_max": goals_max,
                "inference_t_mean": inference_t_mean
            }
        else:
            results = {
                "mean_sucess_r": mean_sucess_r,
                "mean_timeout": mean_timeout,
                "mean_distance_min_s_r": mean_distance_min_s_r,
                "perc_less_21_r": perc_less_21_r,
            }

        print("{:<4} {:<4}".format('Label', 'Number'))
        for k, v in results.items():
            print("{:<4}  {:.4f}".format(k, v))
        results_array = []
        for k, v in results.items():
            print("{:.4f}".format(v))
            results_array.append(v)
        results_array = np.array(results_array)
        epoch_results.append(results_array)
    if epoch>1:
        epoch_results = np.stack(epoch_results, axis=0)
        epoch_results_mean = epoch_results.mean(axis=0)
        print('Mean results over epochs:')
        for v in epoch_results_mean:
            print("{:.4f}".format(v))
        epoch_results_std = epoch_results.std(axis=0)
        print('Std of results over epochs:')
        for v in epoch_results_std:
            print("{:.4f}".format(v))
    return epoch_results_mean, epoch_results_std

def batched_covariance(x):
    """
    Calculate the batched covariance of a given 3D input tensor.

    Args:
        x (torch.Tensor): A 3-dimensional tensor of shape (batch_size, num_features, num_samples).

    Returns:
        torch.Tensor: A 3-dimensional tensor of shape (batch_size, num_features, num_features) representing the batched covariance matrices.
    """
    # Calculate the mean along the num_samples dimension
    mean = x.mean(dim=2, keepdim=True)

    # Subtract the mean from the input tensor
    x_centered = x - mean

    # Calculate the batched covariance
    num_samples = x.size(2)
    covariance = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (num_samples - 1)

    return covariance

def calc_cost_map_cost(trajectory, cost_map_obj, opt_count):
    cost = cost_map_obj.get_cost_from_world_x_y(trajectory[:].cpu().numpy())
    # if opt_count == 4:
    cost = np.where(cost > 97, 100000, cost)

    return cost.sum(0).astype(int)
