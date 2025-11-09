from typing import *
import copy

import torch
from torch import nn

class BYOL(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, moving_average_decay, use_momentum):
        super(BYOL, self).__init__()
        self.online_encoder = net
        self.target_encoder = copy.deepcopy(net)
        feature_dim = self._get_feature_dim()   

        self.online_projector = nn.Sequential(
            nn.Linear(feature_dim, projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size)
        )

        self.target_projector = nn.Sequential(
            nn.Linear(feature_dim, projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size)
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_size, projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size)
        )

        self.moving_average_decay = moving_average_decay
        self.use_momentum = use_momentum

        self._initialize_target_network()

    def _get_feature_dim(self):
        # Dummy forward pass to get feature dimension
        dummy_input = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            features = self.online_encoder(dummy_input)
        return features.size(1)

    def _initialize_target_network(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, view1, view2):
        online_features = self.online_encoder(view1)
        online_projection = self.online_projector(online_features)
        online_prediction = self.online_predictor(online_projection)

        with torch.no_grad():
            target_features = self.target_encoder(view2)
            target_projection = self.target_projector(target_features)

        loss = self._loss_fn(online_prediction, target_projection)
        return loss

    def _loss_fn(self, p, z):
        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=1).mean()

    def update_moving_average(self):
        if not self.use_momentum:
            return  
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = self.moving_average_decay * param_t.data + (1 - self.moving_average_decay) * param_o.data

        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data = self.moving_average_decay * param_t.data + (1 - self.moving_average_decay) * param_o.data
