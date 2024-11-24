import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamQNetwork(nn.Module):
    """
    Neural network for Q-value approximation with separate streams for
    image (occupancy grid) and vector (dynamics) inputs
    """
    def __init__(self, vector_dim: int, grid_size: int, action_dim: int, hidden_dim: int = 128):
        super(DualStreamQNetwork, self).__init__()
        
        # Store dimensions for debugging
        self.vector_dim = vector_dim  # Should be 5 now (orientation + speed + steering + 2*path_error)
        
        # CNN stream for occupancy grid
        self.conv_stream = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten CNN output
        )
        
        # Calculate CNN output size
        self._get_conv_output_size(grid_size)  # This sets self.conv_out_size
        
        # MLP stream for vehicle dynamics
        self.vector_stream = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),  # Input is now 5-dimensional
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(self.conv_out_size + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def _get_conv_output_size(self, grid_size: int) -> int:
        """Calculate the flattened size of the conv layers output"""
        dummy_input = torch.zeros(1, 1, grid_size, grid_size)
        with torch.no_grad():
            conv_out = self.conv_stream(dummy_input)
        self.conv_out_size = int(np.prod(conv_out.shape[1:]))  # Exclude batch dimension
        return self.conv_out_size
    
    def forward(self, grid_input: torch.Tensor, vector_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with both occupancy grid and vehicle dynamics
        Args:
            grid_input: [B, 1, H, W] occupancy grid
            vector_input: [B, 5] vehicle dynamics vector containing:
                         [orientation, speed, steering, path_error[0], path_error[1]]
        Returns:
            [B, action_dim] Q-values for each action
        """
        # Add dimension checks
        if vector_input.shape[1] != self.vector_dim:
            raise ValueError(f"Expected vector_input dimension {self.vector_dim}, got {vector_input.shape[1]}")
            
        # Process occupancy grid
        conv_features = self.conv_stream(grid_input)
        
        # Process vehicle dynamics
        vector_features = self.vector_stream(vector_input)
        
        # Concatenate features and compute Q-values
        combined_features = torch.cat([conv_features, vector_features], dim=1)
        q_values = self.fusion(combined_features)
        
        return q_values