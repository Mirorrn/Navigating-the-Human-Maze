import torch
import torch.nn as nn
from data.utils import dynamic_window, actionXYtoROT
from model.utils import Hist_Encoder, Decoder_TF
import numpy as np
class Decoder_TF(nn.Module):
    def __init__(self, device, input_dim=16):
        super(Decoder_TF, self).__init__()
        self.device = device
        self.d_model = input_dim + 2
        nhead = 2
        dropout = 0.0
        d_hid = 32
        nlayers = 1
        max_t_len = 200
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerDecoderLayer(self.d_model, nhead, d_hid, dropout,layer_norm_eps=0.001)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.dropout = nn.Dropout(p=dropout)
        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        self.fc = nn.Linear(2 * self.d_model, self.d_model)
        self.input_fc = nn.Linear(2, self.d_model)
        self.output_fc = nn.Linear(self.d_model, 4)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=1)

        return pe

    def get_pos(self, num_t, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        # pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def positional_encoding(self, x, t_offset):
        num_t = x.shape[0]
        pos_enc = self.get_pos(num_t, t_offset)
        feat = [x, pos_enc.repeat(1, x.size(1), 1)]
        x = torch.cat(feat, dim=-1)
        x = self.fc(x)
        return self.dropout(x)


    def generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=1)

    
    def forward(self, x, c):
        self.tgt_mask  = self.generate_square_subsequent_mask(x.shape[0])
        x = self.input_fc(x.reshape(-1, x.shape[-1])).view(x.shape[0], x.shape[1], self.d_model)
        x_pos = self.positional_encoding(x, 8)
        x = self.transformer_decoder(x_pos,c, tgt_mask = self.tgt_mask )
        mu, scale = self.output_fc(x.reshape(-1, x.shape[-1])).view(x.shape[0],
                                                                    x.shape[1],
                                                                    4).chunk(2, 2)
        return mu, scale

    def forward_new(self, x, c):
        self.tgt_mask  = self.generate_square_subsequent_mask(x.shape[0])
        x = self.input_fc(x.reshape(-1, x.shape[-1])).view(x.shape[0], x.shape[1], self.d_model)
        x_pos = self.positional_encoding(x, 8)
        x = self.transformer_decoder(x_pos,c, tgt_mask = self.tgt_mask )
        x = self.output_fc(x.reshape(-1, x.shape[-1])).view(x.shape[0],
                                                                    x.shape[1],
                                                                    5) # .chunk(2, 2)
        mu = x[:, :, :2]
        cov_params = x[:, :, 2:]
        L = torch.zeros((x.shape[0], x.shape[1], 2, 2), device=x.device)
        L[..., 0, 0] = torch.exp(cov_params[..., 0])  # Diagonal element (0, 0) is positive
        L[..., 1, 0] = cov_params[..., 1]  # Lower triangular element (1, 0)
        L[..., 1, 1] = torch.exp(cov_params[..., 2])
        return mu, L
class TrajectoryGeneratorGoalIAR(nn.Module):
    def __init__(self,robot_params_dict, dt, device, collision_distance=0.2, obs_len=8,
                 predictions_steps=12, sample_batch=200):
        super(TrajectoryGeneratorGoalIAR, self).__init__()

        self.device = device
        self.robot_params_dict = robot_params_dict
        self.sample_batch = sample_batch
        self.obs_len = obs_len
        self.predictions_steps = predictions_steps
        self.hist_encoder = Hist_Encoder(obs_len, self.device)
        self.decoder = Decoder_TF(self.device)
        self.collision_distance = collision_distance
        # self.critic_imput = torch.zeros([12, self.obs_len, (5 + 1)*sample_batch, 2], device=self.device)
        # self.critic_imput_test = torch.zeros([self.obs_len, (5 + 1) * sample_batch, 2], device=self.device)
        self.dt = dt

    def forward(self, traj_rel, obs_traj_pos,
                nei_index, robot_state, sample_goal, z=None, robotID=None, optimize_latent_space=False, calc_new=True, ar_step_or_DWA=True, mean_pred=False):

        batch = traj_rel.shape[1] #/ 12
        rel_goal = sample_goal - obs_traj_pos[-1]

        if mean_pred:
            noise_sampled = torch.zeros(self.predictions_steps, batch, 2, device=self.device)
        else:
            noise_sampled = torch.randn(self.predictions_steps, batch, 2, device=self.device)

        noise_sampled[:, robotID] = z
        if calc_new:
            num_h = batch // self.sample_batch
            rel_goal = (rel_goal[:num_h]).unsqueeze(dim=0).repeat(8, 1, 1)
            enc_hist = self.hist_encoder(traj_rel[: self.obs_len, :num_h])
            # noise_sampled[:, robotID] = 0.
            context = torch.zeros_like(noise_sampled[:, :num_h])
            enc_hist = torch.cat([rel_goal, enc_hist], dim=-1)
            mu, scale = self.decoder(context, enc_hist)
            scale = torch.clamp(scale, min=-9, max=4)
            self.mu = mu.repeat(1, self.sample_batch, 1)
            self.scale = scale.repeat(1, self.sample_batch, 1)


        output_pred_sampled = self.mu + torch.exp(self.scale) * noise_sampled
        clamped_action = []
        if self.robot_params_dict['use_robot_model'] and ar_step_or_DWA:
            for i in range(self.predictions_steps):
                if optimize_latent_space:
                    u = actionXYtoROT(output_pred_sampled[i, robotID], robot_state, self.dt)
                else:
                    u = z[i]
                robot_state = dynamic_window(robot_state, u,
                                             self.robot_params_dict,
                                             self.dt)
                clamped_action.append(robot_state[:, 3:].unsqueeze(dim=0))
                output_pred_sampled[i, robotID] = robot_state[:, :2] * self.dt
            clamped_action = torch.cat(clamped_action)


        nll = ((output_pred_sampled[:, robotID] - self.mu[:, robotID]) ** 2).sum(-1).sum(0)

        return output_pred_sampled[:self.predictions_steps], clamped_action, nll