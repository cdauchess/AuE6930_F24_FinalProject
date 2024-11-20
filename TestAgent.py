from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
from ReinformentLearning.Environment import RLEnvironment, EpisodeConfig
from ReinformentLearning.RLAgent import AgentConfig, VehicleAgent

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import time

def train_agent(num_episodes: int = 100):
    # Create environment
    bridge = CoppeliaBridge(2)
    config = EpisodeConfig(
        max_steps=200,
        position_range=1.0,
        orientation_range=0.5,
        max_path_error=1.0
    )
    env = RLEnvironment(bridge, config)
    
    # Create agent
    agent_config = AgentConfig()
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
            agent.memory.push(
                agent._preprocess_state(state).squeeze().cpu().numpy(),
                action_idx,  # Store the discrete action index
                reward,
                agent._preprocess_state(next_state).squeeze().cpu().numpy(),
                done
            )
            
            # Train agent
            loss = agent.train(agent_config.batch_size)
            episode_loss += loss
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
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            agent.save(f"agent_checkpoint_{episode+1}.pt")
    
    # Plot training metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    plt.subplot(132)
    plt.plot(episode_losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.subplot(133)
    plt.plot(mean_path_errors)
    plt.title("Mean Path Error")
    plt.xlabel("Episode")
    plt.ylabel("Error")
    
    plt.tight_layout()
    plt.show()
    
    return agent, env

if __name__ == "__main__":
    agent, env = train_agent()