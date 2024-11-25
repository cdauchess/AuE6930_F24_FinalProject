from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, NamedTuple
import numpy as np
import time
import random

from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
from .StateGenerator import VehicleState
from .RewardCalculator import RLReward

@dataclass
class EpisodeConfig:
    """Configuration for RL episodes"""
    max_steps: int = 100
    position_range: float = 1.0
    orientation_range: float = 0.5
    max_path_error: float = 5.0
    time_step: float = 0.05
    render_enabled: bool = True

@dataclass
class EpisodeStats:
    """Statistics for an episode"""
    episode_number: int
    steps: int
    total_reward: float
    mean_path_error: float
    max_path_error: float
    success: bool

class RLEnvironment:
    """RL Environment wrapper for CoppeliaBridge"""
    def __init__(self, bridge: CoppeliaBridge, config: Optional[EpisodeConfig] = None):
        self.bridge = bridge
        self.config = config or EpisodeConfig()
        
        # Episode tracking
        self.episode_count = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.path_errors = []
        
        # Initial pose - we'll get this once during initialization
        self.initial_pose = self._get_initial_pose()
        
        # Reward function
        self.reward_function = RLReward()

    def _get_initial_pose(self) -> Tuple[List[float], List[float]]:
        """Get initial pose from bridge once during initialization"""
        position, orientation = self.bridge.getEgoPoseAbsolute()
        return position, orientation

    def reset(self, randomize: bool = True) -> VehicleState:
        """Reset environment and start new episode"""
        # Stop and restart simulation
        self.bridge.stopSimulation()
        time.sleep(0.5)
        
        self.bridge.setSimStepping(True)
        self.bridge.startSimulation()
        self.bridge.renderState(self.config.render_enabled)
        time.sleep(0.5)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.path_errors = []
        self.episode_count += 1

        # Set vehicle pose
        if randomize:
            self._set_random_pose()
        else:
            self.bridge.setVehiclePose(self.initial_pose[0], self.initial_pose[1])
        
        # Reset vehicle controls
        self.bridge.resetVehicle()
        time.sleep(0.1)
        
        # Return initial state
        return VehicleState.from_bridge(self.bridge)

    def step(self, action: Tuple[float, float]) -> Tuple[VehicleState, float, bool, Dict]:
        """Execute action and return new state, reward, done flag, and info"""
        # Apply action
        speed, steering = action
        self.bridge.setVehicleSpeed(speed)
        self.bridge.setSteering(steering)
        
        # Step simulation
        self.bridge.stepTime()
        self.current_step += 1
        
        # Get new state
        new_state = VehicleState.from_bridge(self.bridge)
        
        # Calculate reward using reward function
        reward = self.reward_function.calculate_reward({
            'speed': new_state.speed,
            'path_error': new_state.path_error[0],  # Using lateral error
            'orientation_error': new_state.path_error[1], # heading_error
            'steering': new_state.steering,
            'collision': self.bridge.checkEgoCollide(new_state.occupancy_grid),
            'success': self._success(new_state)
        })
        
        self.episode_reward += reward
        self.path_errors.append(new_state.path_error[0])
        
        # Check termination
        done = self._is_done(new_state)
        
        # Additional info
        info = {
            'episode': self.episode_count,
            'step': self.current_step,
            'path_error': new_state.path_error[0]
        }
        
        return new_state, reward, done, info

    def _success(self, state: VehicleState) -> bool:
        """Chenck of the episode ended successfully"""
        return self.current_step >= self.config.max_steps
    
    def _is_done(self, state: VehicleState) -> bool:
        """Check if episode should terminate"""
        return (
            self.current_step >= self.config.max_steps or
            abs(state.path_error[0]) > self.config.max_path_error or
            self.bridge.checkEgoCollide(state.occupancy_grid)
        )

    def _set_random_pose(self):
        """Set random initial pose within configured ranges"""
        position = [
            self.initial_pose[0][0] + random.uniform(-self.config.position_range, self.config.position_range),
            self.initial_pose[0][1] + random.uniform(-self.config.position_range, self.config.position_range),
            self.initial_pose[0][2]  # Keep Z constant
        ]
        
        orientation = [
            self.initial_pose[1][0],
            self.initial_pose[1][1],
            self.initial_pose[1][2] + random.uniform(-self.config.orientation_range, self.config.orientation_range)
        ]
        
        self.bridge.setVehiclePose(position, orientation)

    def get_episode_stats(self) -> EpisodeStats:
        """Get statistics for current episode"""
        return EpisodeStats(
            episode_number=self.episode_count,
            steps=self.current_step,
            total_reward=self.episode_reward,
            mean_path_error=np.mean(self.path_errors),
            max_path_error=np.max(self.path_errors),
            success=self.current_step >= self.config.max_steps
        )