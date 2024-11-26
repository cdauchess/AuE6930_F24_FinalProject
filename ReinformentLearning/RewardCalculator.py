from .Configs import RewardConfig

from typing import Dict, Callable, List, Optional
import numpy as np

'''

from dataclasses import dataclass
@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # Vehicle limits
    max_speed: float = 10.0      # m/s
    max_path_error: float = 5.0  # meters
    max_steering: float = 0.5    # radians
    
    # Component weights
    speed_weight: float = 1.0
    path_error_weight: float = 1.0
    steering_weight: float = 0.5
    
    # Penalties and bonuses
    collision_penalty: float = -1.0
    zero_speed_penalty: float = -1.0
    max_path_error_penalty: float = -1.0
    success_reward: float = 1.0

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
        self.active_function = 'simple'
    
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
        reward = (np.exp(-1.0 * normalized_error)) * -self.config.max_path_error_penalty
        return self.config.path_error_weight * reward
    
    def _orientation_reward(self, orientation: float) -> float:
        # TODO: Calculate orientation component errror of reward
        return 0
    
    def _steering_reward(self, steering: float) -> float:
        # TODO: Calculate steering component to encourage smooth control
        return 0
    
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
        """
        Standard reward function combining speed, and path following, steering, and orientation
        """
        #TODO: Update with steer and orient.

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
    
    # Create reward calculator with custom config
    config = RewardConfig(
        max_speed=10.0,
        speed_weight=1.0,
        path_error_weight=1.0,
        steering_weight=0.5
    )
    reward_calc = RLReward(config)
    
    # Plot path error reward function
    errors = np.linspace(-5, 5, 200)
    #errors = np.linspace(0, 10, 200)
    rewards = [reward_calc._path_error_reward(error) for error in errors]
    #rewards = [reward_calc._speed_reward(error) for error in errors]
    
    plt.figure(figsize=(10, 6))
    plt.plot(errors, rewards)
    plt.title('Path Error Reward Function')
    plt.xlabel('Path Error (m/s)')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()
    
    # Example usage
    state = {
        'speed': 5.0,
        'path_error': 0.3,
        'steering': 0.1,
        'collision': False,
        'success': False
    }
    
    # Try different reward functions
    print("\nReward values for example state:")
    for func_name in reward_calc.get_available_functions():
        reward_calc.set_reward_function(func_name)
        reward = reward_calc.calculate_reward(state)
        print(f"{func_name}: {reward:.3f}")