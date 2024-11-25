import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, vector_dim: int, grid_size: int, action_dim: int, hidden_dim: int = 128, 
                 action_bounds: tuple = ((-0.5, 0.5), (0, 10))):
        super(ActorNetwork, self).__init__()
        self.action_bounds = action_bounds
        
        self.conv_stream = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.conv_out_size = self._get_conv_output_size(grid_size)
        
        self.vector_stream = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.conv_out_size + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.steering_head = nn.Linear(hidden_dim, 1)
        self.speed_head = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _get_conv_output_size(self, grid_size: int) -> int:
        dummy_input = torch.zeros(1, 1, grid_size, grid_size)
        with torch.no_grad():
            conv_out = self.conv_stream(dummy_input)
        return int(np.prod(conv_out.shape[1:]))
    
    def forward(self, grid_input: torch.Tensor, vector_input: torch.Tensor) -> torch.Tensor:
        conv_features = self.conv_stream(grid_input)
        vector_features = self.vector_stream(vector_input)
        
        combined = torch.cat([conv_features, vector_features], dim=1)
        features = self.fusion(combined)
        
        steering = torch.tanh(self.steering_head(features))
        acceleration = torch.sigmoid(self.speed_head(features))
        
        steering = steering * (self.action_bounds[0][1] - self.action_bounds[0][0])/2 + \
                  (self.action_bounds[0][1] + self.action_bounds[0][0])/2
        acceleration = acceleration * (self.action_bounds[1][1] - self.action_bounds[1][0]) + \
                      self.action_bounds[1][0]
        
        return torch.cat([steering, acceleration], dim=1)

class CriticNetwork(nn.Module):
    def __init__(self, vector_dim: int, grid_size: int, action_dim: int, hidden_dim: int = 128):
        super(CriticNetwork, self).__init__()
        
        self.conv_stream = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.conv_out_size = self._get_conv_output_size(grid_size)
        
        self.vector_stream = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.action_stream = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.conv_out_size + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _get_conv_output_size(self, grid_size: int) -> int:
        dummy_input = torch.zeros(1, 1, grid_size, grid_size)
        with torch.no_grad():
            conv_out = self.conv_stream(dummy_input)
        return int(np.prod(conv_out.shape[1:]))
    
    def forward(self, grid_input: torch.Tensor, vector_input: torch.Tensor, 
                action: torch.Tensor) -> torch.Tensor:
        conv_features = self.conv_stream(grid_input)
        vector_features = self.vector_stream(vector_input)
        action_features = self.action_stream(action)
        
        combined = torch.cat([conv_features, vector_features, action_features], dim=1)
        return self.fusion(combined)