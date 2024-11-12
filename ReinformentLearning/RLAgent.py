from ReinformentLearning.NeuralNetworks import QNetwork
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class AgentConfig:
    """Configuration for DQN Agent"""
    state_dim: int = 8  # position(3), orientation(1), speed(1), steering(1), path_error(2)
    action_dim: int = 9  # 3 speed levels × 3 steering levels
    hidden_dim: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.99  # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100
    batch_size: int = 4
    target_update: int = 10  # episodes between target network updates

class ReplayBuffer:
    """Experience replay buffer for storing transitions"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        transitions = random.sample(self.buffer, batch_size)
        batch = list(zip(*transitions))
        return [np.array(x) for x in batch]
    
    def __len__(self) -> int:
        return len(self.buffer)

class VehicleAgent:
    """DQN Agent for vehicle control"""
    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = QNetwork(config.state_dim, config.action_dim, 
                                config.hidden_dim).to(self.device)
        self.target_network = QNetwork(config.state_dim, config.action_dim, 
                                     config.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.buffer_size)
        self.epsilon = config.epsilon_start
        
        # Action space discretization
        self.speed_actions = np.array([0.0, 5.0, 10.0])  # m/s
        self.steering_actions = np.array([-0.2, 0.0, 0.2])  # rad
        
    def select_action(self, state: Dict) -> Tuple[float, float, int]:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.config.action_dim)
        else:
            state_tensor = self._preprocess_state(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        # Convert action index to speed and steering
        speed_idx = action_idx // len(self.steering_actions)
        steering_idx = action_idx % len(self.steering_actions)
        return (self.speed_actions[speed_idx], 
                self.steering_actions[steering_idx], 
                action_idx)
    
    def train(self, batch_size: int) -> float:
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch and prepare tensors
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config.gamma * max_next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.config.epsilon_end, 
                         self.epsilon * self.config.epsilon_decay)
    
    def _preprocess_state(self, state: Dict) -> torch.Tensor:
        """Convert state dict to tensor"""
        state_array = np.concatenate([
            state['position'],
            [state['orientation'][2]],  # Only yaw angle
            [state['speed']],
            [state['steering']],
            state['path_error'][:2]  # X and Y error
        ])
        return torch.FloatTensor(state_array).unsqueeze(0).to(self.device)

    def save(self, path: str):
        """Save model weights and training state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }, path)
        
    def load(self, path: str):
        """Load model weights and training state"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.config = checkpoint['config']