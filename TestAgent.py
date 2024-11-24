from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
from ReinformentLearning.Environment import RLEnvironment, EpisodeConfig
from ReinformentLearning.RLAgent import AgentConfig, VehicleAgent

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple

def train_agent(num_episodes: int = 100) -> Tuple[VehicleAgent, RLEnvironment]:
    # Create environment
    bridge = CoppeliaBridge()
    config = EpisodeConfig(
        max_steps=200,
        position_range=1.0,
        orientation_range=0.5,
        max_path_error=1.0,
        render_enabled=False
    )
    env = RLEnvironment(bridge, config)
    
    # Get a sample state to verify dimensions
    initial_state = env.reset()
    _, vector_input = initial_state.get_network_inputs()
    vector_dim = vector_input.shape[0]  # Should be 5 now
    
    # Create agent with correct dimensions
    agent_config = AgentConfig(
        state_dim=vector_dim,  # Should be 5 (orientation + speed + steering + 2*path_error)
        action_dim=9,  # 3 speed levels × 3 steering levels
        hidden_dim=128,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update=10
    )
    agent = VehicleAgent(agent_config)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    mean_path_errors = []
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset(randomize=True)
        episode_reward = 0
        episode_loss = 0
        
        # Episode loop
        while True:
            # Select and perform action
            speed, steering, action_idx = agent.select_action(state)
            next_state, reward, done, _ = env.step((speed, steering))
            
            # Store transition
            agent.store_transition(state, action_idx, reward, next_state, done)
            
            # Train agent
            loss = agent.train(agent_config.batch_size)
            episode_loss += loss if loss is not None else 0
            episode_reward += reward
            
            if done:
                break
                
            state = next_state
        
        # Update target network periodically
        if episode % agent_config.target_update == 0:
            agent.update_target_network()
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Get episode statistics
        stats = env.get_episode_stats()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss)
        mean_path_errors.append(stats.mean_path_error)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Mean Path Error: {stats.mean_path_error:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Loss: {episode_loss:.3f}")
    
    # Plot training metrics
    plot_training_metrics(episode_rewards, episode_losses, mean_path_errors)
    
    return agent, env

def plot_training_metrics(rewards: List[float], losses: List[float], errors: List[float]):
    """
    Plot training metrics
    Args:
        rewards: List of episode rewards
        losses: List of training losses
        errors: List of mean path errors
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    plt.subplot(132)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(133)
    plt.plot(errors)
    plt.title("Mean Path Error")
    plt.xlabel("Episode")
    plt.ylabel("Error (m)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_agent(agent: VehicleAgent, env: RLEnvironment, num_episodes: int = 5):
    """
    Test the trained agent
    Args:
        agent: Trained VehicleAgent
        env: RLEnvironment
        num_episodes: Number of test episodes
    """
    for episode in range(num_episodes):
        state = env.reset(randomize=True)
        episode_reward = 0
        steps = 0
        
        print(f"\nStarting Test Episode {episode + 1}")
        
        while True:
            # Select action (no exploration)
            agent.epsilon = 0
            speed, steering, _ = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step((speed, steering))
            episode_reward += reward
            steps += 1
            
            # Print step information
            if steps % 20 == 0:
                print(f"Step {steps}")
                print(f"  Speed: {speed:.2f} m/s")
                print(f"  Steering: {steering:.2f} rad")
                print(f"  Path Error: {next_state.path_error[0]:.2f} m")
            
            if done:
                break
                
            state = next_state
        
        # Print episode statistics
        print(f"\nTest Episode {episode + 1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Mean Path Error: {np.mean(env.path_errors):.2f}")
        
        # Add delay between episodes
        time.sleep(1)

if __name__ == "__main__":
    # Train agent
    print("Starting training...")
    agent, env = train_agent(num_episodes=100)
    
    # Test agent
    print("\nStarting testing...")
    test_agent(agent, env)