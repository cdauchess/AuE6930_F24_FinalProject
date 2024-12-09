from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, NamedTuple
import numpy as np
import torch

@dataclass
class VehicleAction:
    """Representation of vehicle control actions"""
    steering: float  # steering angle in radians
    acceleration: float  # acceleration in m/s^2

    def to_numpy(self):
        """Convert to numpy array for network processing"""
        return np.array([self.steering, self.acceleration], dtype=np.float32)

    @classmethod
    def from_numpy(cls, array):
        """Create action from numpy array"""
        return cls(steering=float(array[0]), acceleration=float(array[1]))

class VehicleState(NamedTuple):
    """Representation of vehicle state"""
    position: np.ndarray          # [x, y, z] (kept for other uses but not part of network input)
    orientation: float           # yaw angle
    speed: float                # current speed
    steering: float             # steering angle
    path_error: np.ndarray      # [lateral_error, heading_error]
    occupancy_grid: np.ndarray  # binary grid representing obstacles
    distance : float            # Distance traveled in the current step

    @classmethod
    def from_bridge(cls, bridge: CoppeliaBridge):
        """Create state from bridge object"""
        vehicle_state = bridge.getVehicleState()
        path_error, orient_error = bridge.getPathError()
        occupancy_grid = bridge.getOccupancyGrid()
        
        return cls(
            position=np.array(vehicle_state['Position'], dtype=np.float32),
            orientation=vehicle_state['Orientation'],
            speed=vehicle_state['Speed'], # ???
            steering=vehicle_state['Steering'], # -pi/2 to pi/2
            path_error=np.array([path_error, orient_error], dtype=np.float32), # [-5 to 5], [-pi, pi]
            occupancy_grid=np.array(occupancy_grid, dtype=np.float32),
            distance = vehicle_state['distance']
        )
    
    def get_network_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:

        # Process occupancy grid - ensure shape is [C, H, W]
        grid_tensor = torch.FloatTensor(self.occupancy_grid)
        
        # Process vehicle dynamics - create flat vector with relevant state components
        dynamics = np.concatenate([
            [self.orientation],       # 1 value
            [self.speed],            # 1 value
            [self.steering],         # 1 value
            self.path_error.flatten() # 2 values
        ]).astype(np.float32)
        
        dynamics_tensor = torch.FloatTensor(dynamics)
        
        return grid_tensor, dynamics_tensor