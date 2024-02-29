import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(5600, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        
    def forward(self, obs):
        # if isinstance(obs, np.ndarray):
        #     obs = torch.from_numpy(obs)
        if isinstance(obs, tuple) or isinstance(obs, list):
            obs = torch.tensor(obs)
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        # Flatten the input tensor
        obs = obs.view(1, -1)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output