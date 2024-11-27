from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
from ReinformentLearning.Environment import RLEnvironment, EpisodeConfig
from ReinformentLearning.RLAgent import DDPGAgent, DDPGConfig
import time

def test_agent(model_path: str, num_episodes: int = 5):
    # Create environment
    bridge = CoppeliaBridge()
    config = EpisodeConfig(
        max_steps=200,
        position_range=1.0,
        orientation_range=0.5,
        max_path_error=1.0
    )
    env = RLEnvironment(bridge, config)
    
    # Get sample state to verify dimensions
    initial_state = env.reset()
    _, vector_input = initial_state.get_network_inputs()
    vector_dim = vector_input.shape[0]
    
    # Create and load agent
    agent_config = DDPGConfig(
        state_dim=vector_dim,
        action_dim=2,
        hidden_dim=256,
        action_bounds=((-0.5, 0.5), (0, 10))
    )
    agent = DDPGAgent(agent_config)
    agent.load(model_path)
    
    print(f"Starting testing of model: {model_path}")
    for episode in range(num_episodes):
        state = env.reset(randomize=True)
        episode_reward = 0
        steps = 0
        
        print(f"\nStarting Test Episode {episode + 1}")
        
        while True:
            action = agent.select_action(state, add_noise=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if steps % 20 == 0:
                print(f"Step {steps}")
                print(f"  Acceleration: {action.acceleration:.2f} m/s²")
                print(f"  Steering: {action.steering:.2f} rad")
                print(f"  Path Error: {next_state.path_error[0]:.2f} m")
            
            if done:
                break
                
            state = next_state
        
        stats = env.get_episode_stats()
        print(f"\nTest Episode {episode + 1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Mean Path Error: {stats.mean_path_error:.2f}")
        
        time.sleep(1)

if __name__ == "__main__":
    test_agent("agent_trained01.pt")