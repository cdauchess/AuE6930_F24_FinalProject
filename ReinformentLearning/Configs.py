from dataclasses import dataclass

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


@dataclass
class DDPGConfig:
    """Configuration for DDPG Agent"""
    state_dim: int = 5
    action_dim: int = 2  # [steering, speed]
    hidden_dim: int = 256
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    gamma: float = 0.99
    tau: float = 0.001  # For soft target updates
    noise_std: float = 0.1
    buffer_size: int = 1000
    batch_size: int = 128
    action_bounds: tuple = ((-0.5, 0.5), (0, 10))  # ((steering_min, steering_max), (speed_min, speed_max))
