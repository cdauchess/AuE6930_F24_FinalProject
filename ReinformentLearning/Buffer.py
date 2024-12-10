from typing import Tuple, List, Dict
import numpy as np
import torch
from collections import deque, namedtuple
from .VehicleHandler import VehicleState, VehicleAction

# Define a transition tuple for more efficient storage
Transition = namedtuple('Transition', [
    'grid_input', 'vector_input',  # State
    'action',                      # Action
    'reward',                      # Reward
    'next_grid_input', 'next_vector_input',  # Next State
    'done'                         # Done flag
])

class ReplayBuffer:
    """
    Optimized replay buffer with pre-processing and batching capabilities
    """
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state: VehicleState, action: VehicleAction, 
            reward: float, next_state: VehicleState, done: bool):
        """Store preprocessed transition in buffer"""
        # Pre-process state and next_state
        grid_input, vector_input = state.get_network_inputs()
        next_grid_input, next_vector_input = next_state.get_network_inputs()
        
        # Convert action to numpy array
        action_array = action.to_numpy()
        
        # Create transition
        transition = Transition(
            grid_input=grid_input.numpy(),
            vector_input=vector_input.numpy(),
            action=action_array,
            reward=reward,
            next_grid_input=next_grid_input.numpy(),
            next_vector_input=next_vector_input.numpy(),
            done=done
        )
        
        # Store transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions and return tensorized batch"""
        assert len(self.buffer) >= batch_size, "Not enough transitions in buffer"
        
        # Randomly sample indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        # Prepare batch tensors
        grid_inputs = torch.FloatTensor(np.stack([t.grid_input for t in batch])).to(self.device)
        vector_inputs = torch.FloatTensor(np.stack([t.vector_input for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.stack([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.stack([t.reward for t in batch])).unsqueeze(1).to(self.device)
        next_grid_inputs = torch.FloatTensor(np.stack([t.next_grid_input for t in batch])).to(self.device)
        next_vector_inputs = torch.FloatTensor(np.stack([t.next_vector_input for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.stack([t.done for t in batch])).unsqueeze(1).to(self.device)
        
        return (grid_inputs, vector_inputs, actions, rewards, 
                next_grid_inputs, next_vector_inputs, dones)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.position = 0
        
    def get_statistics(self) -> Dict:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "percent_full": 0.0
            }
        
        # Calculate statistics from stored transitions
        rewards = np.array([t.reward for t in self.buffer])
        actions = np.stack([t.action for t in self.buffer])
        
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "percent_full": len(self.buffer) / self.capacity * 100,
            "reward_stats": {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "min": np.min(rewards),
                "max": np.max(rewards)
            },
            "action_stats": {
                "mean": np.mean(actions, axis=0),
                "std": np.std(actions, axis=0)
            }
        }