
from ReinformentLearning.Configs import EpisodeConfig, DDPGConfig
from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
from ReinformentLearning.Environment import RLEnvironment
from ReinformentLearning.RLAgent import DDPGAgent
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_training_metrics(rewards: List[float], losses: List[float], errors: List[float]):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    plt.subplot(132)
    plt.plot(losses)
    plt.title("Actor Loss")
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
    plt.savefig('training_metrics.png')
    plt.close()

def train_agent(num_episodes: int = 10, model_name: str = None):
    # Create environment
    bridge = CoppeliaBridge()
    config = EpisodeConfig()
    
    env = RLEnvironment(bridge, config)
    
    # Get a sample state to verify dimensions
    initial_state = env.reset()
    _, vector_input = initial_state.get_network_inputs()
    vector_dim = vector_input.shape[0]
    
    # Create agent
    agent_config = DDPGConfig(state_dim = vector_dim)
    
    agent = DDPGAgent(agent_config)
    
    if model_name:
        print(f'Loading Agent =: {model_name}')
        agent.load(model_name)

    # Training metrics
    episode_rewards = []
    episode_losses = []
    mean_path_errors = []
    
    print("Starting training...")
    for episode in range(num_episodes):
        state = env.reset(randomize=True)
        episode_reward = 0
        actor_loss = critic_loss = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            
            losses = agent.train(agent_config.batch_size)
            if losses:
                actor_loss, critic_loss = losses
            episode_reward += reward
            
            if done:
                break
                
            state = next_state
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_losses.append(actor_loss)
        mean_path_errors.append(env.get_episode_stats().mean_path_error)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Mean Path Error: {env.get_episode_stats().mean_path_error:.2f}")
            print(f"  Actor Loss: {actor_loss:.3f}")
            print(f"  Critic Loss: {critic_loss:.3f}")
            
        if (episode + 1) % 50 == 0:
            # Save agent and plot metrics
            agent.save("agent_trained.pt")
            plot_training_metrics(episode_rewards, episode_losses, mean_path_errors)
            print('Training Checkpoint Saved!')
    
    # Save agent and plot metrics
    agent.save("agent_trained.pt")
    plot_training_metrics(episode_rewards, episode_losses, mean_path_errors)
    print("Training completed. Model saved as 'agent_trained.pt'")

if __name__ == "__main__":
    train_agent(num_episodes=300, model_name="TrainedAgents/agent_trained07_300.pt")