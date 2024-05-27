
import torch
import d3rlpy
from torch import nn
from model.utils import Pooling_net, Hist_Encoder
from data.utils import prepare_states, cart2pol, pol2cart

class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size, hist, num_agent, device='cuda'):
        super(CustomEncoder, self).__init__()
        self.feature_size = feature_size
        self.hist = hist
        self.num_agent = num_agent
        self.device = device
        self.lstm_hidden_size = 16
        self.obs_len = 8
        self.emb = nn.Linear(2, self.lstm_hidden_size)
        self.fc = nn.Linear(self.lstm_hidden_size*2 + 5, feature_size)
        self.fc1 = nn.Linear(feature_size, feature_size)
        self.hist_encoder = Hist_Encoder(8, self.device)
        self.pl_net = Pooling_net(self.device, self.num_agent, self.num_agent,
                                  h_dim=self.lstm_hidden_size, ar_model=True)


    def forward(self, x):

        with torch.no_grad():
            obs_traj_pos, traj_rel, neigh_index, robot_idx,\
            r_goal, r_pose = prepare_states(x, self.hist, self.num_agent, device=self.device)

            # goal_diff = (r_goal[robot_idx] - obs_traj_pos[-1, robot_idx])
            # v_g, yaw_g = cart2pol(goal_diff[:, 0], goal_diff[:, 1])
            # v_g = torch.clamp(v_g, max=3.0)
            # x_g, y_g = pol2cart(v_g, yaw_g)
            # r_goal = torch.cat([x_g, y_g], dim=-1) + obs_traj_pos[-1, robot_idx]

        batch = obs_traj_pos.shape[1]
        corr = obs_traj_pos[-1].repeat(batch, 1, 1)
        corr_index = (corr.transpose(0, 1) - corr)
        enc_hist = self.hist_encoder(traj_rel[: self.obs_len])[-1]
        pooled_context = self.pl_net(corr_index, neigh_index, enc_hist)
        spatial_emb = torch.relu(self.emb(torch.cat([r_goal - obs_traj_pos[-1, robot_idx]], dim=-1)))
        # h = torch.relu(self.fc(spatial_emb + pooled_context[robot_idx]))
        h = torch.relu(self.fc(torch.cat([spatial_emb, pooled_context[robot_idx], r_pose], dim=-1)))
        return h
#
    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.feature_size
#

class CustomEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size, hist, num_agent, device='cuda'):
        super(CustomEncoderWithAction, self).__init__()
        self.device = device
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.feature_size = feature_size
        self.hist = hist
        self.num_agent = num_agent
        self.lstm_hidden_size = 16
        self.obs_len = 8
        self.hist_encoder = Hist_Encoder(self.obs_len, self.device)
        self.emb = nn.Linear(2 + self.action_size, self.lstm_hidden_size)
        self.fc = nn.Linear(2*self.lstm_hidden_size +5, feature_size)
        self.fc1 = nn.Linear(feature_size, feature_size)
        self.pl_net = Pooling_net(self.device, self.num_agent, self.num_agent,
                                  h_dim=self.lstm_hidden_size, ar_model=True)


    def forward(self, x, action):

        with torch.no_grad():
            obs_traj_pos, traj_rel, neigh_index, robot_idx,\
            r_goal, r_pose = prepare_states(x, self.hist, self.num_agent, device=self.device)

            # goal_diff = (r_goal[robot_idx] - obs_traj_pos[-1, robot_idx])
            # v_g, yaw_g = cart2pol(goal_diff[:, 0], goal_diff[:, 1])
            # v_g = torch.clamp(v_g, max=3.0)
            # x_g, y_g = pol2cart(v_g, yaw_g)
            # r_goal = torch.cat([x_g, y_g], dim=-1) + obs_traj_pos[-1, robot_idx]
        sample_batch = 400
        batch = obs_traj_pos.shape[1]

        corr = obs_traj_pos[-1].repeat(batch, 1, 1)
        corr_index = (corr.transpose(0, 1) - corr)
        enc_hist = self.hist_encoder(traj_rel[: self.obs_len])[-1]
        pooled_context = self.pl_net(corr_index, neigh_index, enc_hist)
        spatial_emb = torch.relu(self.emb(torch.cat([r_goal - obs_traj_pos[-1, robot_idx], action], dim=-1)))
        # h = torch.relu(self.fc(spatial_emb + pooled_context[robot_idx]))
        # h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc(torch.cat([spatial_emb, pooled_context[robot_idx], r_pose], dim=-1)))
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
        return CustomEncoder(observation_shape, self.feature_size, self.hist, self.num_agent, device=self.device)

    def create_with_action(self, observation_shape, action_size, discrete_action=False):
        return CustomEncoderWithAction(observation_shape, action_size, self.feature_size, self.hist, self.num_agent,
                                       device=self.device)

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}



