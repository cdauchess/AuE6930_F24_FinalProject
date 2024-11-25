from .NeuralNetworks import DualStreamQNetwork
from .StateGenerator import VehicleState

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class AgentConfig:
    """Configuration for DQN Agent"""
    state_dim: int = 5  # orientation(1), speed(1), steering(1), path_error(2)
    action_dim: int = 9  # 3 steering levels x 3 speed levels
    hidden_dim: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.99  # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 200000
    batch_size: int = 128
    target_update: int = 10  # episodes between target network updates

class ReplayBuffer:
    """Experience replay buffer for storing transitions"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: 'VehicleState', action: int, reward: float, 
             next_state: 'VehicleState', done: bool):
        """Store transition in buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions
        Returns tuple of (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch_size)
        batch = list(zip(*transitions))
        
        # Keep states and next_states as VehicleState objects
        states = batch[0]
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = batch[3]
        dones = np.array(batch[4])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class VehicleAgent:
    """DQN Agent for vehicle control with dual-stream architecture"""
    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DualStreamQNetwork(
            vector_dim=5,  # orientation(1) + speed(1) + steering(1) + path_error(2)
            grid_size=180,  # Size of occupancy grid
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        self.target_network = DualStreamQNetwork(
            vector_dim=5,
            grid_size=180,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.buffer_size)
        self.epsilon = config.epsilon_start
        
        # Action space discretization
        self.speed_actions = np.array([0.0, 5.0, 10.0])  # m/s
        self.steering_actions = np.array([-0.2, 0.0, 0.2])  # rad
    
    def select_action(self, state: VehicleState) -> Tuple[float, float, int]:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.config.action_dim)
        else:
            grid_input, vector_input = state.get_network_inputs()
            # Add batch dimension and move to device
            grid_input = grid_input.unsqueeze(0).to(self.device)
            vector_input = vector_input.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(grid_input, vector_input)
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
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Process batch of states
        grid_inputs = []
        vector_inputs = []
        next_grid_inputs = []
        next_vector_inputs = []
        
        # Process each state in the batch
        for state in states:
            grid_input, vector_input = state.get_network_inputs()
            grid_inputs.append(grid_input)
            vector_inputs.append(vector_input)
        
        for next_state in next_states:
            next_grid_input, next_vector_input = next_state.get_network_inputs()
            next_grid_inputs.append(next_grid_input)
            next_vector_inputs.append(next_vector_input)
        
        # Convert lists to batched tensors
        grid_inputs = torch.stack(grid_inputs).to(self.device)
        vector_inputs = torch.stack(vector_inputs).to(self.device)
        next_grid_inputs = torch.stack(next_grid_inputs).to(self.device)
        next_vector_inputs = torch.stack(next_vector_inputs).to(self.device)
        
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q = self.q_network(grid_inputs, vector_inputs).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_grid_inputs, next_vector_inputs)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * self.config.gamma * max_next_q
        
        # Compute loss and update
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def store_transition(self, state: VehicleState, action: int, 
                        reward: float, next_state: VehicleState, done: bool):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.config.epsilon_end, 
                         self.epsilon * self.config.epsilon_decay)
    
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