from .Configs import RewardConfig

from typing import Dict, Callable, List, Optional
import numpy as np

'''

from dataclasses import dataclass
@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # Vehicle limits
    max_speed: float = 3.0
    max_path_error: float = 1.0
    max_steering: float = 0.5
    
    # Component weights
    speed_weight: float = 1.0
    path_error_weight: float = 1.0
    steering_weight: float = 1.0
    
    # Penalties and bonuses
    collision_penalty: float = -2.0  # Increased penalty
    zero_speed_penalty: float = -1.0
    max_path_error_penalty: float = -2.0
    success_reward: float = 2.0  # Increased reward for completing longer episodes

'''

class RLReward:
    """Reinforcement Learning reward calculator with multiple reward functions"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # Dictionary of available reward functions
        self.reward_functions = {
            'standard': self._standard_reward,
            'simple': self._simple_reward,
            'smooth': self._smooth_reward
        }
        
        # Default reward function
        self.active_function = 'standard'
    
    def set_reward_function(self, function_name: str) -> bool:
        """
        Set the active reward function
        """
        if function_name in self.reward_functions:
            self.active_function = function_name
            return True
        return False
    
    def calculate_reward(self, state: Dict) -> float:
        """
        Calculate reward based on vehicle state
        """
        # Validate input
        required_keys = {'speed', 'path_error', 'orientation_error', 'steering', 'collision', 'success'}
        if not all(key in state for key in required_keys):
            raise ValueError(f"State must contain keys: {required_keys}")
        
        return self.reward_functions[self.active_function](state)
    
    def _speed_reward(self, speed: float) -> float:
        """Calculate speed component of reward"""
        if speed <= 0:
            return self.config.zero_speed_penalty
        return self.config.speed_weight * (speed / self.config.max_speed) ** 2
        
    def _path_error_reward(self, error: float) -> float:
        """Calculate path error component of reward using smooth function"""
        normalized_error = abs(error)
        if normalized_error >= self.config.max_path_error:
            return self.config.max_path_error_penalty
        
        # Smooth exponential decay
        reward = (np.exp(-5.0 * normalized_error))
        return self.config.path_error_weight * reward
    
    def _orientation_reward(self, orientation_error: float) -> float:
        """
        Calculate orientation reward based on heading error to path trajectory
        orientation_error: Difference between vehicle heading and path trajectory
        """
        # Normalize to [-pi, pi]
        normalized_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))
        
        # Exponential reward that peaks at zero error and decays with larger errors
        # exp(-x^2/2) gives smooth falloff and is always positive
        reward = np.exp(-0.5 * (normalized_error ** 2))
    
        # Scale reward
        return self.config.steering_weight * (2 * reward - 1)

    
    def _steering_reward(self, steering: float) -> float:
        """Calculate steering reward that encourages smooth, appropriate turning"""
        normalized_steering = abs(steering) / self.config.max_steering
        
        # Small penalty for very aggressive steering (> 80% of max)
        aggressive_steering_penalty = -0.5 * max(0, normalized_steering - 0.8)**2
        
        # Moderate penalty for rapid steering changes near limits
        stability_reward = -0.2 * (normalized_steering**4)
        
        return aggressive_steering_penalty + stability_reward + 0.1
    
    def _simple_reward(self, state: Dict) -> float:
        """
        Simplified reward function focusing mainly on path following and moving forward.
        """
        reward = 0.0
        
        # Core components
        reward += self._speed_reward(state['speed'])
        reward += self._path_error_reward(state['path_error'])
        
        # Penalties and bonuses
        if state['collision']:
            reward += self.config.collision_penalty
        
        if state['success']:
            reward += self.config.success_reward
            
        return reward

    def _standard_reward(self, state: Dict) -> float:
        reward = 0.0
        
        # Core components
        reward += self._speed_reward(state['speed'])
        reward += self._path_error_reward(state['path_error'])
        reward += self._orientation_reward(state['orientation_error'])
        reward += self._steering_reward(state['steering'])
        
        # Penalties
        if state['collision']:
            reward += self.config.collision_penalty
        if state['success']:
            reward += self.config.success_reward
            
        return reward

    
    def _smooth_reward(self, state: Dict) -> float:
        
        # TODO: Reward function emphasizing smooth control

        path_reward = self._path_error_reward(state['path_error'])
        steering_reward = self._steering_reward(state['steering']) * 2.0  # Increased weight on smooth steering
        speed_reward = self._speed_reward(state['speed']) * 0.5  # Reduced weight on speed
        
        reward = path_reward + steering_reward + speed_reward
        
        if state['collision']:
            reward += self.config.collision_penalty
        if state['success']:
            reward += self.config.success_reward
            
        return reward

    def get_available_functions(self) -> List[str]:
        """Get list of available reward functions"""
        return list(self.reward_functions.keys())

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    reward_calc = RLReward()
    
    # Create subplots for different reward components
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Path Error Reward
    errors = np.linspace(-reward_calc.config.max_path_error, reward_calc.config.max_path_error, 200)
    path_rewards = [reward_calc._path_error_reward(error) for error in errors]
    ax1.plot(errors, path_rewards)
    ax1.set_title('Path Error Reward')
    ax1.set_xlabel('Path Error (m)')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # 2. Speed Reward
    speeds = np.linspace(0, reward_calc.config.max_speed, 200)
    speed_rewards = [reward_calc._speed_reward(speed) for speed in speeds]
    ax2.plot(speeds, speed_rewards)
    ax2.set_title('Speed Reward')
    ax2.set_xlabel('Speed (m/s)')
    ax2.set_ylabel('Reward')
    ax2.grid(True)
    
    # 3. Orientation Reward
    orientations = np.linspace(-np.pi, np.pi, 200)
    orientation_rewards = [reward_calc._orientation_reward(orient) for orient in orientations]
    ax3.plot(orientations, orientation_rewards)
    ax3.set_title('Orientation Reward')
    ax3.set_xlabel('Orientation Error (rad)')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    
    # 4. Steering Reward
    steerings = np.linspace(-reward_calc.config.max_steering, reward_calc.config.max_steering, 200)
    steering_rewards = [reward_calc._steering_reward(steer) for steer in steerings]
    ax4.plot(steerings, steering_rewards)
    ax4.set_title('Steering Reward')
    ax4.set_xlabel('Steering Angle (rad)')
    ax4.set_ylabel('Reward')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Test different reward functions with example states
    test_states = [
        {
            'speed': 0.0,
            'path_error': 0.0,
            'orientation_error': 0.0,
            'steering': 0.0,
            'collision': False,
            'success': False,
            'description': 'Ideal State'
        },
        {
            'speed': 1.0,
            'path_error': 0.3,
            'orientation_error': 0.1,
            'steering': 0.1,
            'collision': False,
            'success': False,
            'description': 'Normal Operation'
        },
        {
            'speed': 1.5,
            'path_error': 2.0,
            'orientation_error': 0.5,
            'steering': 0.4,
            'collision': False,
            'success': False,
            'description': 'Large Error State'
        }
    ]
    
    print("\nReward values for different states:")
    for state in test_states:
        print(f"\n{state['description']}:")
        for func_name in reward_calc.get_available_functions():
            reward_calc.set_reward_function(func_name)
            reward = reward_calc.calculate_reward(state)
            print(f"{func_name:8}: {reward:.3f}")