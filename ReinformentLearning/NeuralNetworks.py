import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ResidualLinear(nn.Module):
    def __init__(self, hidden_dim: int):
        super(ResidualLinear, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.linear1(x)))
        out = self.ln2(self.linear2(out))
        out += residual
        return F.relu(out)

class ActorNetwork(nn.Module):
    def __init__(self, vector_dim: int, grid_size: int, action_dim: int, hidden_dim: int = 256,
                 action_bounds: tuple = ((-0.5, 0.5), (0, 1))):
        super(ActorNetwork, self).__init__()
        self.action_bounds = action_bounds
        
        # Improved CNN with residual connections
        self.conv_stream = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # Calculate conv output size
        dummy_input = torch.zeros(1, 4, grid_size, grid_size)
        conv_out = self.conv_stream(dummy_input)
        self.conv_out_size = conv_out.shape[1]
        
        # Improved vector processing
        self.vector_stream = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Improved fusion network
        self.fusion = nn.Sequential(
            nn.Linear(self.conv_out_size + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads with improved initialization
        self.steering_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
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
    def __init__(self, vector_dim: int, grid_size: int, action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        # Improved CNN for critic
        self.conv_stream = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # Calculate conv output size
        dummy_input = torch.zeros(1, 4, grid_size, grid_size)
        conv_out = self.conv_stream(dummy_input)
        self.conv_out_size = conv_out.shape[1]
        
        # Improved vector processing
        self.vector_stream = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Improved action processing
        self.action_stream = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualLinear(hidden_dim)
        )
        
        # Final layers with better initialization
        self.fusion = nn.Sequential(
            nn.Linear(self.conv_out_size + hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for final layer
        if isinstance(m, nn.Linear) and m.out_features == 1:
            nn.init.uniform_(m.weight, -3e-3, 3e-3)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -3e-3, 3e-3)
    
    def forward(self, grid_input: torch.Tensor, vector_input: torch.Tensor, 
                action: torch.Tensor) -> torch.Tensor:
        conv_features = self.conv_stream(grid_input)
        vector_features = self.vector_stream(vector_input)
        action_features = self.action_stream(action)
        
        combined = torch.cat([conv_features, vector_features, action_features], dim=1)
        return self.fusion(combined)