from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge

import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import time

@dataclass
class EpisodeConfig:
    """Configuration for RL episodes"""
    max_steps: int = 100
    position_range: float = 1.0
    orientation_range: float = 0.5
    max_path_error: float = 5.0
    time_step: float = 0.05

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
        
        # Store initial state
        self._initial_position = self.bridge._sim.getObjectPosition(
            self.bridge._egoVehicle, self.bridge._world)
        self._initial_orientation = self.bridge._sim.getObjectOrientation(
            self.bridge._egoVehicle, self.bridge._world)

    def reset(self, randomize: bool = True) -> Dict:
        """
        Reset environment and start new episode
        Returns: Initial state dictionary
        """
        # Stop current simulation
        self.bridge.stopSimulation()
        
        # Wait a small amount of time to ensure proper cleanup
        time.sleep(0.1)
        
        # Reset simulation settings
        self.bridge._sim.setStepping(True)
        
        # Start new simulation
        self.bridge.startSimulation()
        
        # Wait for simulation to stabilize
        time.sleep(0.1)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.path_errors = []
        self.episode_count += 1

        # Set initial state
        if randomize:
            position = self._get_random_position()
            orientation = self._get_random_orientation()
        else:
            position = self._initial_position
            orientation = self._initial_orientation
            
        # Set position and orientation
        self.bridge.setVehiclePose(position, orientation)
        
        # Reset vehicle controls
        self.bridge.resetVehicle()
        
        # Wait for physics to stabilize
        time.sleep(0.1)
        
        # Return initial observation
        return self._get_observation()

    def step(self, action: Tuple[float, float]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action and return (state, reward, done, info)
        Args:
            action: Tuple of (speed, steering)
        """
        # Apply action
        speed, steering = action
        self.bridge.setSpeed(speed)
        self.bridge.setSteering(steering)
        
        # Step simulation
        self.bridge.stepTime()
        self.current_step += 1
        
        # Get new state and calculate reward
        new_state = self._get_observation()
        reward = self._calculate_reward(new_state)
        self.episode_reward += reward
        
        # Check if episode is done
        done = self._is_done(new_state)
        
        # Get additional info
        info = {
            'episode': self.episode_count,
            'step': self.current_step,
            'path_error': new_state['path_error']
        }
        
        return new_state, reward, done, info

    def _get_observation(self) -> Dict:
        """Get current state observation"""
        vehicle_state = self.bridge.getVehicleState()
        path_error, orient_error = self.bridge.getPathError(self.bridge.activePath)
        
        # Store path error for statistics
        self.path_errors.append(np.linalg.norm(path_error))
        
        return {
            'position': vehicle_state['Position'],
            'orientation': vehicle_state['Orientation'],
            'speed': vehicle_state['Speed'],
            'steering': vehicle_state['Steering'],
            'path_error': path_error,
            'orient_error': orient_error
        }

    def _calculate_reward(self, state: Dict) -> float:
        """Calculate reward for current state"""
        path_error = state['path_error']
        steering = state['steering']
        
        # Calculate different reward components
        distance_error = np.linalg.norm(path_error)
        path_reward = -distance_error
        steering_penalty = -0.1 * abs(steering/self.bridge._maxSteerAngle)
        
        return path_reward + steering_penalty

    def _is_done(self, state: Dict) -> bool:
        """Check if episode should terminate"""
        # Check termination conditions
        max_steps_reached = self.current_step >= self.config.max_steps
        path_error = np.linalg.norm(state['path_error'])
        off_track = path_error > self.config.max_path_error
        
        return max_steps_reached or off_track

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

    def _get_random_position(self) -> List[float]:
        """Generate random initial position"""
        return [
            self._initial_position[0] + random.uniform(
                -self.config.position_range, self.config.position_range),
            self._initial_position[1] + random.uniform(
                -self.config.position_range, self.config.position_range),
            self._initial_position[2]  # Keep Z constant
        ]

    def _get_random_orientation(self) -> List[float]:
        """Generate random initial orientation"""
        return [
            self._initial_orientation[0],
            self._initial_orientation[1],
            self._initial_orientation[2] + random.uniform(
                -self.config.orientation_range, self.config.orientation_range)
        ]