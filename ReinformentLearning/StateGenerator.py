from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, NamedTuple
import numpy as np
import torch

class VehicleState(NamedTuple):
    """Representation of vehicle state"""
    position: np.ndarray          # [x, y, z] (kept for other uses but not part of network input)
    orientation: float           # yaw angle
    speed: float                # current speed
    steering: float             # steering angle
    path_error: np.ndarray      # [lateral_error, heading_error]
    occupancy_grid: np.ndarray  # binary grid representing obstacles

    @classmethod
    def from_bridge(cls, bridge: CoppeliaBridge):
        """Create state from bridge object"""
        vehicle_state = bridge.getVehicleState()
        path_error, orient_error = bridge.getPathError()
        occupancy_grid = bridge.getOccupancyGrid()
        
        return cls(
            position=np.array(vehicle_state['Position'], dtype=np.float32),
            orientation=vehicle_state['Orientation'],
            speed=vehicle_state['Speed'],
            steering=vehicle_state['Steering'],
            path_error=np.array([path_error, orient_error], dtype=np.float32),
            occupancy_grid=np.array(occupancy_grid, dtype=np.float32)
        )
    
    def get_network_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert state to network inputs (grid and vector components)
        Returns:
            Tuple of (occupancy_grid_tensor, dynamics_tensor)
        """
        # Process occupancy grid - ensure shape is [1, H, W]
        grid_tensor = torch.FloatTensor(self.occupancy_grid).unsqueeze(0)
        
        # Process vehicle dynamics - create flat vector with relevant state components
        dynamics = np.concatenate([
            [self.orientation],       # 1 value
            [self.speed],            # 1 value
            [self.steering],         # 1 value
            self.path_error.flatten() # 2 values
        ]).astype(np.float32)
        
        dynamics_tensor = torch.FloatTensor(dynamics)
        
        return grid_tensor, dynamics_tensor