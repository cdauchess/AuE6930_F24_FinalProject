
from ReinformentLearning.Configs import EpisodeConfig, DDPGConfig
from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
from ReinformentLearning.Environment import RLEnvironment
from ReinformentLearning.RLAgent import DDPGAgent
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_rewards(episode_rewards: List[Dict]):
    """Create separate plots for total reward and reward components"""
    # Extract components
    episodes = range(len(episode_rewards))
    total_rewards = [r['total'] for r in episode_rewards]
    path_rewards = [r['path'] for r in episode_rewards]
    speed_rewards = [r['speed'] for r in episode_rewards]
    steering_rewards = [r['steering'] for r in episode_rewards]
    tracking_rewards = [r['tracking'] for r in episode_rewards]
    speed_consistency_rewards = [r['speed_consistency'] for r in episode_rewards]
    damping_rewards = [r['damping'] for r in episode_rewards]
    
    plt.figure(figsize=(15, 10))
    
    # Total reward plot
    plt.subplot(2, 1, 1)
    plt.plot(episodes, total_rewards, 'b-', label='Total Reward')
    plt.title('Total Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(-1000, 1000)
    plt.grid(True)
    plt.legend()
    
    # Reward components plot
    plt.subplot(2, 1, 2)
    plt.plot(episodes, path_rewards, label='Path')
    plt.plot(episodes, speed_rewards, label='Speed')
    plt.plot(episodes, steering_rewards, label='Steering')
    plt.plot(episodes, tracking_rewards, label='Tracking')
    plt.plot(episodes, speed_consistency_rewards, label='Speed Consistency')
    plt.plot(episodes, damping_rewards, label='Damping')
    plt.title('Reward Components')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(-1000, 1000)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_losses(losses: List[Dict]):
    """Create plot for actor and critic losses"""
    episodes = range(len(losses))
    actor_losses = [loss.get('actor', 0) for loss in losses]
    critic_losses = [loss.get('critic', 0) for loss in losses]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, actor_losses, 'g-', label='Actor Loss')
    plt.plot(episodes, critic_losses, 'r-', label='Critic Loss')
    plt.title('Network Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics(errors: List[float], speeds: List[float], distances: List[float]):
    """Create plot for path error, mean speed, and distance traveled"""
    episodes = range(len(errors))
    
    plt.figure(figsize=(15, 5))
    
    # Path Error
    plt.subplot(1, 3, 1)
    plt.plot(episodes, errors, 'b-')
    plt.title('Mean Path Error')
    plt.xlabel('Episode')
    plt.ylabel('Error (m)')
    plt.grid(True)
    
    # Mean Speed
    plt.subplot(1, 3, 2)
    plt.plot(episodes, speeds, 'g-')
    plt.title('Mean Speed')
    plt.xlabel('Episode')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    
    # Distance Traveled
    plt.subplot(1, 3, 3)
    plt.plot(episodes, distances, 'm-')
    plt.title('Distance Traveled')
    plt.xlabel('Episode')
    plt.ylabel('Distance (m)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
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
    episode_rewards = []  # Will store dictionaries of reward components
    episode_losses = []
    mean_path_errors = []
    mean_speeds = []
    distances_traveled = []
    
    print("Starting training...")
    for episode in range(num_episodes):
        state = env.reset(randomize=True)
        episode_reward_dict = {
            'total': 0, 'path': 0, 'speed': 0, 'steering': 0,
            'tracking': 0, 'speed_consistency': 0, 'damping': 0,
            'collision': 0, 'success': 0
        }
        actor_loss = critic_loss = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward_dict, done, _ = env.step(action)
            
            # Store transition with total reward
            agent.store_transition(state, action, reward_dict['total'], next_state, done)
            
            losses = agent.train(agent_config.batch_size)
            if losses:
                actor_loss, critic_loss = losses
            
            # Accumulate reward components
            for key in episode_reward_dict:
                episode_reward_dict[key] += reward_dict[key]
            
            if done:
                break
                
            state = next_state
        
        # Store metrics
        stats = env.get_episode_stats()
        episode_rewards.append(episode_reward_dict)
        episode_losses.append({'actor': actor_loss, 'critic': critic_loss})
        mean_path_errors.append(stats.mean_path_error)
        mean_speeds.append(stats.mean_speed)
        distances_traveled.append(stats.distance_traveled)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}")
            print(f"  Total Reward: {episode_reward_dict['total']:.2f}")
            print(f"  Mean Path Error: {stats.mean_path_error:.2f}")
            print(f"  Mean Speed: {stats.mean_speed:.2f}")
            print(f"  Distance: {stats.distance_traveled:.2f}")
            print(f"  Actor Loss: {actor_loss:.3f}")
            print(f"  Critic Loss: {critic_loss:.3f}")
            
        if (episode + 1) % 50 == 0:
            # Save agent and plot metrics
            agent.save(f"agent_trained_{episode+1}.pt")
            plot_rewards(episode_rewards)
            plot_losses(episode_losses)
            plot_metrics(mean_path_errors, mean_speeds, distances_traveled)
            print('Training Checkpoint Saved!')
    
    # Save final model and plot final metrics
    agent.save("agent_trained.pt")
    plot_rewards(episode_rewards)
    plot_losses(episode_losses)
    plot_metrics(mean_path_errors, mean_speeds, distances_traveled)
    print("Training completed. Model saved as 'agent_trained.pt'")

if __name__ == "__main__":
    #train_agent(num_episodes=300, model_name="TrainedAgents/agent_trained07_300.pt")
    train_agent(num_episodes=800)