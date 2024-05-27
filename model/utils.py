import torch
import torch.nn as nn
import numpy as np


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class Pooling_net(nn.Module):
    def __init__(
            self, device, num_agents, embedding_dim=32, h_dim=32,
            activation='relu', batch_norm=False, dropout=0.0, ar_model = False
    ):
        super(Pooling_net, self).__init__()
        self.device = device
        self.ar_model = ar_model
        self.num_agents = num_agents + 1
        self.h_dim = h_dim
        self.bottleneck_dim = h_dim
        self.embedding_dim = embedding_dim
        if self.ar_model:
            self.mlp_pre_dim = embedding_dim + h_dim * 2
            self.mlp_pre_pool_dims = [self.mlp_pre_dim, 64, self.bottleneck_dim]
        else:
            self.mlp_pre_dim = embedding_dim
            self.mlp_pre_pool_dims = [self.mlp_pre_dim, 16, self.bottleneck_dim]

        self.attn = nn.Linear(self.bottleneck_dim, 1)
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            self.mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def forward(self, corr_index, nei_index, lstm_state):
        self.N = corr_index.shape[0]
        if self.ar_model:
            hj_t = lstm_state[nei_index[:, 1]]
            hi_t = lstm_state[nei_index[:, 0]]
            r_t = self.spatial_embedding(corr_index[nei_index[:, 0], nei_index[:, 2]])
            mlp_h_input = torch.cat((r_t, hj_t, hi_t), 1)
        else:
            r_t = self.spatial_embedding(corr_index[nei_index[:, 0], nei_index[:, 1]])
            mlp_h_input = r_t
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)
        # Message Passing
        H = torch.full((self.N, self.num_agents, self.bottleneck_dim), -np.Inf, device=self.device,
                       dtype=curr_pool_h.dtype)
        H[nei_index[:, 0], nei_index[:, 2]] = curr_pool_h
        pool_h = H.max(1)[0]
        pool_h[pool_h == -np.Inf] = 0.
        return pool_h

    def forward_old(self, corr_index, nei_index, nei_num, lstm_state, curr_pos_abs, plot_att=False):
        self.N = corr_index.shape[0]
        hj_t = lstm_state.unsqueeze(0).expand(self.N, self.N, self.h_dim)
        hi_t = lstm_state.unsqueeze(1).expand(self.N, self.N, self.h_dim)
        nei_index_t = nei_index.view((-1))
        corr_t = corr_index.reshape((self.N * self.N, -1))
        r_t = self.spatial_embedding(corr_t[nei_index_t > 0])
        mlp_h_input = torch.cat((r_t, hj_t[nei_index > 0], hi_t[nei_index > 0]), 1)
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)
        # Message Passing
        H = torch.full((self.N * self.N, self.bottleneck_dim), -np.Inf, device=torch.device("cuda"),
                       dtype=curr_pool_h.dtype)
        H[nei_index_t > 0] = curr_pool_h
        pool_h = H.view(self.N, self.N, -1).max(1)[0]
        pool_h[pool_h == -np.Inf] = 0.
        return pool_h, (0, 0, 0), 0
class Hist_Encoder(nn.Module):
    def __init__(self, hist_len, device, input_dim = 2):
        super(Hist_Encoder, self).__init__()
        self.d_model = 16
        self.hist_len = hist_len
        self.device = device
        nhead = 2
        dropout = 0.0
        d_hid = 32
        nlayers = 1
        max_t_len = 200
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # self.pred_hidden2pos = nn.Linear(16, 2 * 2)
        self.dropout = nn.Dropout(p=dropout)
        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        self.fc = nn.Linear(2 * self.d_model, self.d_model, device=self.device)
        self.input_fc = nn.Linear(input_dim, self.d_model, device=self.device)

        self.scr_mask = self.generate_square_subsequent_mask(self.hist_len)

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
    
    def forward(self, x, c=None):
        # test = x.numpy()
        x = self.input_fc(x.reshape(-1, x.shape[-1])).view(x.shape[0], x.shape[1], self.d_model)
        x_pos = self.positional_encoding(x, 0)
        x_enc = self.transformer_encoder(x_pos, mask=self.scr_mask)
        return x_enc

class Decoder_TF(nn.Module):
    def __init__(self, device, input_dim=16):
        super(Decoder_TF, self).__init__()
        self.device = device
        self.d_model = input_dim
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