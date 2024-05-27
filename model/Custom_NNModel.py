
import torch
import d3rlpy
from torch import nn
from model.utils import Pooling_net, Hist_Encoder
from data.utils import prepare_states, cart2pol, pol2cart
import matplotlib.pyplot as plt
import numpy as np


def add_collision_probability(robot_rel_vecs):
    """ Add Collision Probablity feature and sort by x_indicies """
    dist_mat = torch.sqrt(robot_rel_vecs[:,:,0]**2 + robot_rel_vecs[:,:,1]**2)
    col_prob2 = torch.where(torch.logical_and(0. != dist_mat, dist_mat != 0.),1 * (1 - torch.sigmoid((dist_mat - 0.55)*10.0)), 0.0) 
    col_prob2 = col_prob2.unsqueeze(-1)
    robot_rel_vecs_col_prob = torch.cat([robot_rel_vecs, col_prob2],dim=2)
    
    return robot_rel_vecs_col_prob


class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size, hist, num_agent, device):
        super(CustomEncoder, self).__init__()
        self.feature_size = feature_size
        self.hist = hist
        self.num_agent = num_agent
        self.device = device
        self.lstm_hidden_size = 256
        self.obs_len = 8
        self.emb = nn.Linear(2, self.lstm_hidden_size).to(device=self.device)
        self.fc = nn.Linear(2+self.num_agent*3, feature_size).to(device=self.device)
        self.fc1 = nn.Linear(5*2, feature_size).to(device=self.device)
        self.fc2 = nn.Linear(feature_size*1 + 5 + (self.num_agent + 1)*16, feature_size).to(device=self.device)
        self.dropout_fc = nn.Dropout(0.1).to(device=self.device)
        self.dropout_fc2 = nn.Dropout(0.1).to(device=self.device)
        self.hist_encoder = Hist_Encoder(self.obs_len, self.device, input_dim=4).to(device=self.device)



    def forward(self, x):

        x = x.to(device=self.device)
        with torch.no_grad():
            obs_traj_pos, traj_rel, _, robot_idx,\
            r_goal, r_pose, inx_agents_in_seq = prepare_states(x, self.hist, self.num_agent, full_return=True, device=self.device)

            padded_cur_agents_obs = torch.zeros([x.shape[0] * (self.num_agent + 1), 2], device=self.device)
            padded_cur_agents_obs[inx_agents_in_seq] = obs_traj_pos[-1]
            padded_cur_agents_obs = padded_cur_agents_obs.reshape([-1, self.num_agent + 1, 2])
            padded_cur_agents_obs = padded_cur_agents_obs[:,1:]
            sorted_rel_vecs = add_collision_probability(padded_cur_agents_obs)

            padded_hist = torch.zeros([x.shape[0] * (self.num_agent + 1), 16], device=self.device)
            goal_robot_rel = r_goal - obs_traj_pos[-1, robot_idx]

        hist_to_encode =  torch.cat([traj_rel[: self.obs_len], obs_traj_pos], dim = -1)
        enc_hist = self.hist_encoder(hist_to_encode)[-1]

        padded_hist[inx_agents_in_seq] = enc_hist
        padded_hist = padded_hist.reshape([-1, self.num_agent + 1, 16])
        padded_hist = padded_hist.reshape([-1, (self.num_agent + 1)*16])

        h = torch.relu(self.fc(torch.cat([goal_robot_rel, sorted_rel_vecs.reshape(-1,(self.num_agent)*3)], dim=-1)))
        h = self.dropout_fc(h)
        h = torch.relu(self.fc2(torch.cat([h,r_pose, padded_hist], dim=-1)))
        h = self.dropout_fc2(h)
        return h
#
    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.feature_size
#

class CustomEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size, hist, num_agent, device):
        super(CustomEncoderWithAction, self).__init__()
        self.device = device
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.feature_size = feature_size
        self.hist = hist
        self.num_agent = num_agent
        self.lstm_hidden_size = 256
        self.obs_len = 8
        self.hist_encoder = Hist_Encoder(self.obs_len, self.device, input_dim=4).to(device=self.device)
        self.emb = nn.Linear(2 + self.action_size, self.lstm_hidden_size).to(device=self.device)
        self.fc = nn.Linear(2+self.num_agent*3+self.action_size, feature_size).to(device=self.device)
        self.fc1 = nn.Linear(5*2, feature_size).to(device=self.device)
        self.fc2 = nn.Linear(feature_size*1 + 5 + (self.num_agent + 1)*16, feature_size).to(device=self.device)
        self.dropout_fc = nn.Dropout(0.1).to(device=self.device)
        self.dropout_fc2 = nn.Dropout(0.1).to(device=self.device)



    def forward(self, x, action):

        x = x.to(device=self.device)
        with torch.no_grad():
            obs_traj_pos, traj_rel, _, robot_idx,\
            r_goal, r_pose, inx_agents_in_seq = prepare_states(x, self.hist, self.num_agent, full_return=True, device=self.device)


            padded_cur_agents_obs = torch.zeros([x.shape[0] * (self.num_agent + 1), 2], device=self.device)
            padded_cur_agents_obs[inx_agents_in_seq] = obs_traj_pos[-1]
            padded_cur_agents_obs = padded_cur_agents_obs.reshape([-1, self.num_agent + 1, 2])
            padded_cur_agents_obs = padded_cur_agents_obs[:,1:]
            sorted_rel_vecs = add_collision_probability(padded_cur_agents_obs)

            padded_hist = torch.zeros([x.shape[0] * (self.num_agent + 1), 16], device=self.device)
            goal_robot_rel = r_goal - obs_traj_pos[-1, robot_idx]

        hist_to_encode =  torch.cat([traj_rel[: self.obs_len], obs_traj_pos], dim = -1)
        enc_hist = self.hist_encoder(hist_to_encode)[-1]

        padded_hist[inx_agents_in_seq] = enc_hist
        padded_hist = padded_hist.reshape([-1, self.num_agent + 1, 16])
        padded_hist = padded_hist.reshape([-1, (self.num_agent + 1)*16])

        h = torch.relu(self.fc(torch.cat([goal_robot_rel, sorted_rel_vecs.reshape(-1,(self.num_agent)*3), action], dim=-1)))
        h = self.dropout_fc(h)
        h = torch.relu(self.fc2(torch.cat([h,r_pose, padded_hist], dim=-1)))
        h = self.dropout_fc2(h)
        return h

    def get_feature_size(self):
      return self.feature_size
#
#
class CustomEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"

    def __init__(self, feature_size, hist, num_agent, device):
        self.feature_size = feature_size
        self.hist = hist
        self.num_agent = num_agent
        self.device = device

    def create(self, observation_shape):
        model_CustomEncoder = CustomEncoder(observation_shape, self.feature_size, self.hist, self.num_agent, device=self.device)
        return model_CustomEncoder

    def create_with_action(self, observation_shape, action_size, discrete_action=False):
        mode_CustomEncoderWithAction = CustomEncoderWithAction(observation_shape, action_size, self.feature_size, self.hist, self.num_agent,
                                       device=self.device)
        return mode_CustomEncoderWithAction

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}
    
    def get_type(self):
        return "custom"


