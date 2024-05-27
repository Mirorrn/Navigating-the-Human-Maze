import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import Pooling_net, Hist_Encoder

from data.utils import dynamic_window, actionXYtoROT, GaußNLL

class TrajectoryGeneratorAR_goal(nn.Module):
    def __init__(self, num_agent, robot_params_dict, dt, collision_distance=0.2, obs_len=8,
                 predictions_steps=12, sample_batch=200, device='cuda'):
        super(TrajectoryGeneratorAR_goal, self).__init__()
        self.num_agent = num_agent
        self.robot_params_dict = robot_params_dict
        self.sample_batch = sample_batch
        self.collision_distance = collision_distance
        self.obs_len = obs_len
        self.pred_len = predictions_steps
        traj_lstm_input_size = 2
        rela_embed_size = 16
        traj_lstm_hidden_size = 16
        self.device = device
        self.inputLayer_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 16, rela_embed_size)
        self.pl_net = Pooling_net(self.device, self.num_agent, h_dim=traj_lstm_hidden_size, ar_model=True)
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size
        self.hist_encoder = Hist_Encoder(obs_len, self.device)
        self.pred_lstm_model = nn.LSTMCell(rela_embed_size, 16)
        self.pred_hidden2pos1 = nn.Linear(traj_lstm_hidden_size + 2, traj_lstm_hidden_size)
        self.pred_hidden2pos2 = nn.Linear(traj_lstm_hidden_size, 2 * 2)
        self.dt = dt
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, traj_rel, obs_traj_pos,
                nei_index, robot_state, sample_goal, z=None, robotID=None, optimize_latent_space=False, mean_pred=True, proposed_robot_action=None,):

        pred_traj_rel = []

        batch = obs_traj_pos.shape[1]
        num_h = batch // self.sample_batch
        rel_goal = sample_goal - obs_traj_pos[-1]
        enc_hist = self.hist_encoder(traj_rel[: self.obs_len])[-1]
        pred_lstm_hidden = enc_hist

        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden, device=self.device)
        output = traj_rel[self.obs_len - 1]
        lstm_state_context = torch.zeros_like(pred_lstm_hidden, device=self.device)
        curr_pos_abs = obs_traj_pos[-1]
        clamped_action = []
        nll = 0
        for i in range(self.pred_len):

            input_cat = torch.cat([lstm_state_context, output], dim=-1).detach()
            input_embedded = self.dropout(F.relu(self.inputLayer_decoder(input_cat)))
            lstm_state = self.pred_lstm_model(
                input_embedded, (pred_lstm_hidden, pred_lstm_c_t)
            )
            pred_lstm_hidden = lstm_state[0]
            pred_lstm_c_t = lstm_state[1]

            curr_pos_abs_res = curr_pos_abs.reshape(self.sample_batch, num_h, 2)  # now here repeate just the 6
            dist_mat = curr_pos_abs_res.unsqueeze(dim=1).expand(self.sample_batch, num_h, num_h, 2)
            dist_mat = (dist_mat.transpose(1, 2) - dist_mat) #  + 10
            dist_mat = dist_mat.reshape(-1, num_h, 2)

            lstm_state_context = self.pl_net(dist_mat, nei_index, pred_lstm_hidden)
            concat_output = pred_lstm_hidden + lstm_state_context
            h = F.relu(self.pred_hidden2pos1(torch.cat([concat_output, rel_goal], dim=-1)))
            mu, scale = self.pred_hidden2pos2(h).chunk(2, 1)
            scale = torch.clamp(scale, min=-9, max=4)
            if mean_pred:
                sample_noise = torch.zeros_like(scale)
            else:
                sample_noise = torch.randn_like(scale)
                sample_noise[robotID] = 0.0

            if optimize_latent_space:
                sample_noise[robotID] = z[i] # effective only for optimization in latent space


            output_pred = mu + torch.exp(scale) * sample_noise
            if proposed_robot_action is not None:
                output_pred[robotID] = proposed_robot_action[i]


            if self.robot_params_dict['use_robot_model']:

                if optimize_latent_space:
                    u = actionXYtoROT(output_pred[robotID], robot_state, self.dt)
                else:
                    u = z[i]
                robot_state = dynamic_window(robot_state, u,
                                             self.robot_params_dict,
                                             self.dt)
                clamped_action.append(robot_state[:, 3:].unsqueeze(dim=0))
                output_pred[robotID] = robot_state[:, :2] * self.dt

            nll = nll + ((mu[robotID] - output_pred[robotID]) ** 2).sum(-1)  # nll with variance 1


            curr_pos_abs = (curr_pos_abs + output_pred).detach()
            pred_traj_rel += [output_pred]
            output = output_pred

        if self.robot_params_dict['use_robot_model']:
            clamped_action = torch.cat(clamped_action)
        else:
            clamped_action = z
        pred_traj_rel = torch.stack(pred_traj_rel)
     #   nll = ((pred_traj_abs - mu_test)**2).sum(0).sum(-1)
    # dfd
        return pred_traj_rel, clamped_action, nll# [robotID]
