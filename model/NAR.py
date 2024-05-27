import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import Pooling_net, Hist_Encoder

from data.utils import dynamic_window, actionXYtoROT, GaußNLL

# class Pooling_net(nn.Module):
#     def __init__(
#             self, embedding_dim=32, h_dim=32,
#             activation='relu', batch_norm=False, dropout=0.0
#     ):
#         super(Pooling_net, self).__init__()
#         self.h_dim = h_dim
#         self.bottleneck_dim = h_dim
#         self.embedding_dim = embedding_dim
#
#         self.mlp_pre_dim = embedding_dim + h_dim * 2
#         self.mlp_pre_pool_dims = [self.mlp_pre_dim, 64, self.bottleneck_dim]
#         self.attn = nn.Linear(self.bottleneck_dim, 1)
#         self.spatial_embedding = nn.Linear(2, embedding_dim)
#         self.mlp_pre_pool = make_mlp(
#             self.mlp_pre_pool_dims,
#             activation=activation,
#             batch_norm=batch_norm,
#             dropout=dropout)
#
#     def forward(self, corr_index, nei_index, lstm_state):
#         self.N = corr_index.shape[0]
#         hj_t = lstm_state[nei_index[:, 1]]
#         hi_t = lstm_state[nei_index[:, 0]]
#         r_t = self.spatial_embedding(corr_index[nei_index[:, 0], nei_index[:, 2]])
#         mlp_h_input = torch.cat((r_t, hj_t, hi_t), 1)
#         curr_pool_h = self.mlp_pre_pool(mlp_h_input)
#         # Message Passing
#         H = torch.full((self.N, 6, self.bottleneck_dim), -np.Inf, device=torch.device("cuda"),
#                        dtype=curr_pool_h.dtype)
#         H[nei_index[:, 0], nei_index[:, 2]] = curr_pool_h
#         pool_h = H.max(1)[0]
#         # pool_h = H[:,0]
#         pool_h[pool_h == -np.Inf] = 0.
#         return pool_h

class TrajectoryGeneratorAR(nn.Module):
    def __init__(self, num_agent, robot_params_dict, dt, collision_distance=0.2, obs_len=8,
                 predictions_steps=12, sample_batch=200, device='cuda'):
        super(TrajectoryGeneratorAR, self).__init__()
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
        self.hist_encoder = Hist_Encoder(obs_len,self.device)
        self.pred_lstm_model = nn.LSTMCell(rela_embed_size, 16)
        self.pred_hidden2pos = nn.Linear(traj_lstm_hidden_size, 2 * 2)
        self.dt = dt
        self.dropout = nn.Dropout(p=0.0)


    def forward(self, traj_rel, obs_traj_pos,
                nei_index, robot_state, z=None, robotID=None,
                optimize_latent_space=False, mean_pred=True,
                proposed_robot_action=None, robot_pred=True):

        pred_traj_rel = []
        mu_robot_list, scale_robot_list = [], []

        batch = obs_traj_pos.shape[1]
        num_h = batch // self.sample_batch

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

            distance = curr_pos_abs.reshape(self.sample_batch, num_h, 2)  # now here repeate just the 6 agents
            distance = distance.unsqueeze(dim=1).expand(self.sample_batch, num_h, num_h, 2)
            distance = (distance.transpose(1, 2) - distance)
            distance = distance.reshape(-1, num_h, 2)

            lstm_state_context = self.pl_net(distance, nei_index, pred_lstm_hidden)
            concat_output = pred_lstm_hidden + lstm_state_context
            mu, scale = self.pred_hidden2pos(concat_output).chunk(2, 1)
            scale = torch.clamp(scale, min=-9, max=4)
            if mean_pred:
                sample_noise = torch.zeros_like(scale)
            else:
                sample_noise = torch.randn_like(scale) * 0.3
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

            nll = nll + ((mu - output_pred) ** 2).sum(-1)  # GaußNLL(mu, scale, output_pred)

            curr_pos_abs = (curr_pos_abs + output_pred).detach()
            pred_traj_rel += [output_pred]
            output = output_pred

        if self.robot_params_dict['use_robot_model']:
            clamped_action = torch.cat(clamped_action)
        pred_traj_rel = torch.stack(pred_traj_rel)

        return pred_traj_rel, clamped_action, nll[robotID]
