import torch
import torch.nn as nn
from torch.nn import functional as F


def glorot_init(p):
    if isinstance(p, nn.Linear):
        nn.init.xavier_normal_(p.weight.data, gain=1.)
        nn.init.zeros_(p.bias)


class tanh_gaussian_actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size, log_std_min, log_std_max):
        super(tanh_gaussian_actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dims)
        self.log_std = nn.Linear(hidden_size, action_dims)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # init the networks
        self.apply(glorot_init)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp the log std
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        # the reparameterization trick
        # return mean and std
        return (mean, torch.exp(log_std))


class Critic(nn.Module):
    """
    construct a classifier C(s,a) -> [0,1]
    """

    def __init__(self, obs_dim, action_dim, hidden_size, loss_type):
        super(Critic, self).__init__()
        self.loss_type = loss_type
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)

        self.apply(glorot_init)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.q(x)
        if self.loss_type == 'c':
            q = torch.sigmoid(q)
        return q


