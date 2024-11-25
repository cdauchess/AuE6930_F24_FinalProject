from collections import deque
import numpy as np
import random
from typing import Tuple, List
from .VehicleHandler import VehicleState, VehicleAction

class ReplayBuffer:
    """
    A circular buffer for storing and sampling transitions for DDPG.
    Specifically handles continuous action spaces.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, 
            state: VehicleState,
            action: VehicleAction,
            reward: float,
            next_state: VehicleState,
            done: bool):

        # Store transition in buffer 
        action_array = action.to_numpy()
        self.buffer.append((state, action_array, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[List[VehicleState], 
                                              List[VehicleAction],
                                              np.ndarray, # Rewards
                                              List[VehicleState],
                                              np.ndarray]: # Dones
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert actions back to VehicleAction objects
        actions = [VehicleAction.from_numpy(a) for a in actions]
        
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return list(states), actions, rewards, list(next_states), dones
    
    def __len__(self) -> int:
        """Get current number of transitions in buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Clear all transitions from buffer"""
        self.buffer.clear()

    def get_statistics(self) -> dict:
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "percent_full": 0.0
            }
        
        # Calculate basic stats
        rewards = [transition[2] for transition in self.buffer]
        actions = np.array([transition[1] for transition in self.buffer])
        
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "percent_full": len(self.buffer) / self.capacity * 100,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_action": np.mean(actions, axis=0),
            "std_action": np.std(actions, axis=0)
        }