from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, NamedTuple
import numpy as np
import time
import random

from .Configs import EpisodeConfig, EpisodeStats
from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
from .VehicleHandler import VehicleState, VehicleAction
from .RewardCalculator import RLReward

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
        self.vehicle_speed = []
        self.distance_traveled = 0
        
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
        self.bridge.startSimulation(True)
        self.bridge.renderState(self.config.render_enabled)
        time.sleep(0.5)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.path_errors = []
        self.vehicle_speed = []
        self.distance_traveled = 0
        self.episode_count += 1

        # Set vehicle pose
        if randomize:
            self.bridge.setInitPosition(startPos=np.random.rand())
            #self._set_random_pose()
        else:
            self.bridge.setVehiclePose(self.initial_pose[0], self.initial_pose[1])
        
        # Reset vehicle controls
        self.bridge.resetVehicle()
        time.sleep(0.1)
        
        # Return initial state
        return VehicleState.from_bridge(self.bridge)

    def step(self, action: VehicleAction) -> Tuple[VehicleState, float, bool, Dict]:
        """Execute action and return new state, reward, done flag, and info"""
        
        # Apply action
        self.bridge.setMotion(action.acceleration)
        self.bridge.setSteering(action.steering)
        
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
        self.vehicle_speed.append(new_state.speed)
        self.distance_traveled += new_state.distance
        
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
        """Episode succeeds if path completed within error bounds consistently"""
        # Require at least 600 steps (75% of max) with good performance
        return (
            self.current_step >= 600 and
            abs(state.path_error[0]) < self.config.max_path_error * 0.2 and
            abs(state.path_error[1]) < 0.1  # Check orientation error
        )
    
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
            success=self.current_step >= self.config.max_steps,
            mean_speed = np.mean(self.vehicle_speed),
            distance_traveled= self.distance_traveled
        )