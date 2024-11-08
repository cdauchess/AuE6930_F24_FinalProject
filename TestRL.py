from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
import time
import math

def test_episode_functionality():
    # Initialize bridge
    bridge = CoppeliaBridge(2)
    
    # Test multiple episodes
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}")
        
        # Start episode with randomization
        initial_state = bridge.startEpisode(randomize=True)
        print(f"Initial position: {initial_state['Position']}")
        print(f"Initial orientation: {initial_state['Orientation']}")
        
        # Run episode
        done = False
        step = 0
        while not done:
            # Simple test policy: constant speed and oscillating steering
            speed = 5.0
            steering = 0.2 * math.sin(step * 0.1)
            
            # Step episode
            new_state, done = bridge.stepEpisode(speed, steering)
            
            # Print progress every 100 steps
            if step % 100 == 0:
                path_error, _ = bridge.getPathError(bridge.activePath)
                print(f"Step {step}, Path Error: {path_error}")
            
            step += 1
        
        print(f"Episode {episode + 1} finished after {step} steps")
        stats = bridge.getEpisodeStats()
        print(f"Episode stats: {stats}")
        
        # Small delay between episodes
        time.sleep(1)
    
    bridge.stopSimulation()

if __name__ == "__main__":
    test_episode_functionality()