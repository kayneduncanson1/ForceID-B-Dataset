import torch
import torch.nn as nn
import math
from Utils import set_seed

seed = 42
g = set_seed(seed)


# Two-layer fully-connected neural network:
class TwoF(nn.Module):

    def __init__(self, in_features, fc1_out, out_features):
        super(TwoF, self).__init__()

        self.in_features = in_features
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.fc1 = nn.Sequential(nn.Linear(self.in_features, self.fc1_out), nn.BatchNorm1d(num_features=self.fc1_out),
                                 nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):

        out = inputs.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


# This generates the positional encoding for inputs to a transformer encoder layer:
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# Neural network with three convolutional layers, one transformer encoder layer and two fully-connected layers
# (respectively):
class ThrCOneTTwoF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, conv_k, conv_p,
                 pool_k, pool_s, pool_p, len_seq,
                 trans_nhead, trans_act, trans_dim_ff,
                 fc1_out, out_features):
        super(ThrCOneTTwoF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.conv_k = conv_k
        self.conv_p = conv_p
        self.pool_k = pool_k
        self.pool_s = pool_s
        self.pool_p = pool_p
        self.len_seq = len_seq
        self.trans_nhead = trans_nhead
        self.trans_act = trans_act
        self.trans_dim_ff = trans_dim_ff
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=self.conv_k, padding=self.conv_p),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=self.pool_k, stride=self.pool_s, padding=self.pool_p))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=self.conv_k, padding=self.conv_p),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=self.pool_k, stride=self.pool_s, padding=self.pool_p))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=self.conv_k, padding=self.conv_p),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=self.pool_k, stride=self.pool_s, padding=self.pool_p))

        self.pos_encoder = PositionalEncoding(self.nc3, self.len_seq)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.nc3,
                                                        nhead=self.trans_nhead,
                                                        activation=self.trans_act,
                                                        dim_feedforward=self.trans_dim_ff)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(self.trans_dim_ff, self.fc1_out)
        self.bn_elu = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(-1, inputs.size(0), self.nc3)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.bn_elu(out)
        out = self.fc2(out)

        return out


# Class for the Siamese variant of a given neural network (embeddingnet). Used when single_network = False in
# data_analysis.py:
class SiameseNet(nn.Module):

    def __init__(self, embeddingnet):

        super(SiameseNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, input1, input2, cuda):

        features1 = self.embeddingnet(input1, cuda)
        features2 = self.embeddingnet(input2, cuda)

        return features1 + features2

        # Other example operations that could be used for feature fusion:
        # return torch.max(features1, features2)
        # return (features1 + features2) / 2
        # return torch.min(features1, features2)
        # return features1 - features2
        # return features1 / features2
        # Note that the below option doubles the number of output features relative to the out_features variable value
        # specified at the top of data_analysis.py:
        # return torch.cat((features1, features2), dim=1)

