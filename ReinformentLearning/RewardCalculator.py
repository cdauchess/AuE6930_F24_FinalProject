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
    
    # Penalties and bonuses
    collision_penalty: float = -100.0  # Increased penalty
    zero_speed_penalty: float = -10.0
    max_path_error_penalty: float = -10.0
    success_reward: float = 10.0  # Increased reward for completing longer episodes


'''

class RLReward:
    """Reinforcement Learning reward calculator with multiple reward functions"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # Store previous values for rate-of-change calculations
        self._prev_steering = 0.0
        self._prev_path_error = 0.0
        self._prev_speed = 0.0
        self._error_history = []
        self._max_history = 10  # Number of past errors to track
        
        # Dictionary of available reward functions
        self.reward_functions = {
            'standard': self._standard_reward,
            'simple': self._simple_reward,
            'smooth': self._smooth_reward
        }
        
        # Default reward function
        self.active_function = 'standard'
    
    def set_reward_function(self, function_name: str) -> bool:
        """Set the active reward function"""
        if function_name in self.reward_functions:
            self.active_function = function_name
            return True
        return False
    
    def calculate_reward(self, state: Dict) -> Dict:
        """Calculate reward based on vehicle state"""
        required_keys = {'speed', 'path_error', 'orientation_error', 'steering', 'collision', 'success'}
        if not all(key in state for key in required_keys):
            raise ValueError(f"State must contain keys: {required_keys}")
        
        # Update error history
        self._error_history.append(state['path_error'])
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)
        
        reward_dict = self.reward_functions[self.active_function](state)
        
        # Update previous values for next iteration
        self._prev_steering = state['steering']
        self._prev_path_error = state['path_error']
        self._prev_speed = state['speed']
        
        return reward_dict
    
    def _speed_reward(self, speed: float) -> float:
        """Calculate speed component of reward"""
        if speed <= 0.1:
            return self.config.zero_speed_penalty
        return (speed / self.config.max_speed) ** 2
    
    def _path_error_reward(self, error: float) -> float:
        """Calculate path error component of reward using smooth function"""
        normalized_error = abs(error)
        if normalized_error >= self.config.max_path_error:
            return self.config.max_path_error_penalty
        
        # Smooth exponential decay
        reward = (np.exp(-2.0 * normalized_error))
        return reward
    
    def _orientation_reward(self, orientation_error: float) -> float:
        """Calculate orientation reward based on heading error"""
        reward = np.exp(-1.5 * (orientation_error ** 2))
        return 2 * reward - 1
    
    def _steering_reward(self, steering: float) -> float:
        """Calculate steering reward that encourages smooth, appropriate turning"""
        normalized_steering = abs(steering) / self.config.max_steering
        aggressive_steering_penalty = -1 * max(0, normalized_steering - 0.5)**2
        stability_reward = -0.5 * (normalized_steering**4)
        return aggressive_steering_penalty + stability_reward
    
    def _steering_smoothness_reward(self, steering: float) -> float:
        """Calculate reward for smooth steering changes"""
        steering_rate = abs(steering - self._prev_steering)
        # Penalize rapid steering changes with exponential penalty
        smoothness_penalty = 1.25 - np.exp(2.0 * steering_rate)
        return smoothness_penalty
    
    def _path_tracking_consistency_reward(self, path_error: float) -> float:
        """Calculate reward for consistent path tracking"""
        if len(self._error_history) < 2:
            return 0.0
            
        # Calculate error derivative (rate of change)
        error_derivative = abs(path_error - self._prev_path_error)
        
        # Calculate error variance over recent history
        error_variance = np.var(self._error_history) if len(self._error_history) > 1 else 0
        
        # Penalize both rapid error changes and sustained oscillations
        consistency_reward = (
            -1.0 * error_derivative -  # Penalty for rapid error changes
            2.0 * error_variance    # Penalty for sustained oscillations
        )
        
        return consistency_reward
    
    def _speed_consistency_reward(self, speed: float) -> float:
        """Calculate reward for maintaining consistent speed"""
        speed_change = abs(speed - self._prev_speed)
        
        # Penalize rapid speed changes
        consistency_penalty = 1.0 - np.exp(speed_change)
        return consistency_penalty
    
    def _cross_track_damping_reward(self, path_error: float, orientation_error: float) -> float:
        """Calculate reward that encourages damped convergence to path"""
        # Combine position and orientation errors to detect overshooting
        # If error and orientation have opposite signs, the vehicle might be overcorrecting
        error_sign = np.sign(path_error)
        orientation_sign = np.sign(orientation_error)
        
        damping_factor = 1.0
        if error_sign != orientation_sign:
            # Vehicle is turning in the direction that could lead to overshooting
            damping_factor = -0.5
        
        return damping_factor * np.exp(-2.0 * abs(path_error))
    
    def _simple_reward(self, state: Dict) -> float:
        """Simplified reward function focusing mainly on path following"""
        reward = 0.0
        reward += self._speed_reward(state['speed'])
        reward += self._path_error_reward(state['path_error'])
        
        if state['collision']:
            reward += self.config.collision_penalty
        if state['success']:
            reward += self.config.success_reward
        return reward
    
    def _standard_reward(self, state: Dict) -> float:
        """Standard reward function with basic components"""
        reward = 0.0
        speed = self._speed_reward(state['speed'])
        path = self._path_error_reward(state['path_error'])
        orientation = self._orientation_reward(state['orientation_error'])
        steering = self._steering_reward(state['steering'])
        reward = speed+path+steering+orientation
        
        collision = 0
        success = 0
        if state['collision']:
            reward += self.config.collision_penalty
            collision = self.config.collision_penalty
        if state['success']:
            reward += self.config.success_reward
            success = self.config.success_reward
            
        return {
            'total': reward,
            'path': path,
            'speed': speed,
            'steering': steering,
            'tracking': 0,
            'speed_consistency': 0,
            'damping': 0,
            'collision': collision,
            'success': success
        }
    
    def _smooth_reward(self, state: Dict) -> Dict:
        """Enhanced reward function returning components"""
        # Calculate individual components
        path_reward = self._path_error_reward(state['path_error'])
        speed_reward = self._speed_reward(state['speed'])
        steering_reward = self._steering_smoothness_reward(state['steering'])
        tracking_reward = self._path_tracking_consistency_reward(state['path_error'])
        speed_consistency = self._speed_consistency_reward(state['speed'])
        damping_reward = self._cross_track_damping_reward(
            state['path_error'], 
            state['orientation_error']
        )
        
        # Critical penalties
        collision_reward = self.config.collision_penalty if state['collision'] else 0
        success_reward = self.config.success_reward if state['success'] else 0
        
        # Calculate total reward
        total_reward = (path_reward + speed_reward + steering_reward + 
                    tracking_reward + speed_consistency + damping_reward + 
                    collision_reward + success_reward)
        
        return {
            'total': total_reward,
            'path': path_reward,
            'speed': speed_reward,
            'steering': steering_reward,
            'tracking': tracking_reward,
            'speed_consistency': speed_consistency,
            'damping': damping_reward,
            'collision': collision_reward,
            'success': success_reward
        }
    
    def get_available_functions(self) -> List[str]:
        """Get list of available reward functions"""
        return list(self.reward_functions.keys())


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    reward_calc = RLReward()

    # Create figure with subplots for all reward components
    fig = plt.figure(figsize=(20, 15))
    
    # Basic reward components (2x3 grid)
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    ax4 = plt.subplot(3, 3, 4)
    ax5 = plt.subplot(3, 3, 5)
    ax6 = plt.subplot(3, 3, 6)
    ax7 = plt.subplot(3, 3, 7)
    ax8 = plt.subplot(3, 3, 8)
    
    # 1. Path Error Reward
    errors = np.linspace(-reward_calc.config.max_path_error, reward_calc.config.max_path_error, 200)
    path_rewards = [reward_calc._path_error_reward(error) for error in errors]
    ax1.plot(errors, path_rewards, 'b-', linewidth=2)
    ax1.set_title('Path Error Reward')
    ax1.set_xlabel('Path Error (m)')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # 2. Speed Reward
    speeds = np.linspace(0, reward_calc.config.max_speed, 200)
    speed_rewards = [reward_calc._speed_reward(speed) for speed in speeds]
    ax2.plot(speeds, speed_rewards, 'g-', linewidth=2)
    ax2.set_title('Speed Reward')
    ax2.set_xlabel('Speed (m/s)')
    ax2.set_ylabel('Reward')
    ax2.grid(True)
    
    # 3. Orientation Reward
    orientations = np.linspace(-np.pi, np.pi, 200)
    orientation_rewards = [reward_calc._orientation_reward(orient) for orient in orientations]
    ax3.plot(orientations, orientation_rewards, 'r-', linewidth=2)
    ax3.set_title('Orientation Reward')
    ax3.set_xlabel('Orientation Error (rad)')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    
    # 4. Basic Steering Reward
    steerings = np.linspace(-reward_calc.config.max_steering, reward_calc.config.max_steering, 200)
    steering_rewards = [reward_calc._steering_reward(steer) for steer in steerings]
    ax4.plot(steerings, steering_rewards, 'm-', linewidth=2)
    ax4.set_title('Basic Steering Reward')
    ax4.set_xlabel('Steering Angle (rad)')
    ax4.set_ylabel('Reward')
    ax4.grid(True)
    
    # 5. Steering Smoothness Reward
    steering_changes = np.linspace(0, 0.5, 200)  # Range of steering changes
    reward_calc._prev_steering = 0.0  # Reset for visualization
    smoothness_rewards = [reward_calc._steering_smoothness_reward(change) for change in steering_changes]
    ax5.plot(steering_changes, smoothness_rewards, 'c-', linewidth=2)
    ax5.set_title('Steering Smoothness Reward')
    ax5.set_xlabel('Steering Change (rad)')
    ax5.set_ylabel('Reward')
    ax5.grid(True)
    
    # 6. Path Tracking Consistency
    error_changes = np.linspace(-1.0, 1.0, 200)  # Range of error changes

    # Good behavior (smooth following)
    reward_calc._error_history = [0.1, 0.08, 0.06, 0.05, 0.04]  # Gradually converging
    reward_calc._prev_path_error = 0.04  # Reset for visualization
    consistency_rewards = [reward_calc._path_tracking_consistency_reward(change) for change in error_changes]
    ax6.plot(error_changes, consistency_rewards, 'g-', linewidth=2, label='Smooth Following')

    # Bad behavior (oscillating)
    reward_calc._error_history = [0.2, -0.15, 0.25, -0.2, 0.3]  # Zigzagging
    reward_calc._prev_path_error = 0.3  # Reset for visualization
    consistency_rewards = [reward_calc._path_tracking_consistency_reward(change) for change in error_changes]
    ax6.plot(error_changes, consistency_rewards, 'r-', linewidth=2, label='Oscillating')

    ax6.set_title('Path Tracking Consistency Reward')
    ax6.set_xlabel('Path Error Change (m)')
    ax6.set_ylabel('Reward')
    ax6.grid(True)
    ax6.legend(loc='upper right')  # Add legend to upper right corner
    
    # 7. Speed Consistency Reward
    speed_changes = np.linspace(0, 1.0, 200)  # Range of speed changes
    reward_calc._prev_speed = 1.0  # Reset for visualization
    speed_consistency_rewards = [reward_calc._speed_consistency_reward(1.0 + change) for change in speed_changes]
    ax7.plot(speed_changes, speed_consistency_rewards, 'k-', linewidth=2)
    ax7.set_title('Speed Consistency Reward')
    ax7.set_xlabel('Speed Change (m/s)')
    ax7.set_ylabel('Reward')
    ax7.grid(True)
    
    # 8. Cross Track Damping Reward
    errors = np.linspace(-1.0, 1.0, 200)
    damping_rewards = [reward_calc._cross_track_damping_reward(error, error/2) for error in errors]
    ax8.plot(errors, damping_rewards, color='orange', linewidth=2)
    ax8.set_title('Cross Track Damping Reward')
    ax8.set_xlabel('Path Error (m)')
    ax8.set_ylabel('Reward')
    ax8.grid(True)
    

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
            'description': 'Ideal (Stopped) State',
            'error_history': [0.0, 0.0, 0.0, 0.0, 0.0]  # Perfect tracking
        },
        {
            'speed': 1.0,
            'path_error': 0.3,
            'orientation_error': 0.1,
            'steering': 0.1,
            'collision': False,
            'success': False,
            'description': 'Normal Operation',
            'error_history': [0.35, 0.33, 0.31, 0.3, 0.3]  # Gradual improvement
        },
        {
            'speed': 1.5,
            'path_error': 2.0,
            'orientation_error': 0.5,
            'steering': 0.4,
            'collision': False,
            'success': False,
            'description': 'Large Error State',
            'error_history': [1.8, 1.85, 1.9, 1.95, 2.0]  # Growing error
        },
        {
            'speed': 0.8,
            'path_error': 0.2,
            'orientation_error': -0.15,
            'steering': -0.2,
            'collision': False,
            'success': False,
            'description': 'Oscillating State',
            'error_history': [-0.3, 0.25, -0.2, 0.15, -0.2]  # Clear oscillation
        }
    ]

    print("\nReward values for different states:")
    headers = ["State", "Simple", "Standard", "Smooth"]
    print(f"\n{headers[0]:<20} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12}")
    print("-" * 56)

    for state in test_states:
        # Set the error history for this state
        reward_calc._error_history = state['error_history']
        reward_calc._prev_path_error = state['error_history'][-2]  # Second to last value
        
        rewards = []
        for func_name in reward_calc.get_available_functions():
            reward_calc.set_reward_function(func_name)
            reward = reward_calc.calculate_reward(state)
            rewards.append(f"{reward:8.3f}")
        
        print(f"{state['description']:<20} {rewards[0]:<12} {rewards[1]:<12} {rewards[2]:<12}")