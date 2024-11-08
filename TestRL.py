from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
from ReinformentLearning.Environment import RLEnvironment, EpisodeConfig
import numpy as np
import time

def test_rl_environment():
    # Create bridge and environment
    bridge = CoppeliaBridge(2)
    
    # Create custom configuration
    config = EpisodeConfig(
        max_steps=200,
        position_range=1.0,
        orientation_range=0.5,
        max_path_error=1.0
    )
    
    env = RLEnvironment(bridge, config)
    
    # Run test episodes
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}")
        print("Resetting environment...")
        
        # Reset environment
        state = env.reset(randomize=True)
        print("Reset complete.")
        print(f"Initial state: {state}")
        print(f"Simulation running: {bridge._isRunning}")
        
        # Run episode
        done = False
        while not done:
            # Simple test policy
            action = (10.0, 0.2 * np.sin(env.current_step * 0.1))
            
            # Step environment
            new_state, reward, done, info = env.step(action)
            
            # Print progress every 50 steps
            if env.current_step % 50 == 0:
                print(f"Step {info['step']}, Path Error: {new_state['path_error']}")
                print(f"Current Speed: {new_state['speed']}")
                print(f"Current Steering: {new_state['steering']}")
        
        # Get episode statistics
        stats = env.get_episode_stats()
        print(f"\nEpisode {stats.episode_number} finished:")
        print(f"  Steps: {stats.steps}")
        print(f"  Total Reward: {stats.total_reward:.2f}")
        print(f"  Mean Path Error: {stats.mean_path_error:.2f}")
        print(f"  Success: {stats.success}")
        
        # Add delay between episodes
        print("Waiting before next episode...")
        time.sleep(1)
    
    bridge.stopSimulation()

if __name__ == "__main__":
    test_rl_environment()