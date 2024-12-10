from .Configs import DDPGConfig
from .VehicleHandler import VehicleState, VehicleAction
from .NeuralNetworks import ActorNetwork, CriticNetwork
from .Buffer import ReplayBuffer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple
import copy


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration"""
    def __init__(self, size: int, mu: float = 0., theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = copy.copy(self.mu)
    
    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

class DDPGAgent:
    """DDPG Agent with optimized training process"""
    def __init__(self, config: DDPGConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks (same as before)
        self.actor = ActorNetwork(
            vector_dim=config.state_dim,
            grid_size=180,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            action_bounds=config.action_bounds
        ).to(self.device)
        
        self.target_actor = copy.deepcopy(self.actor)
        
        self.critic = CriticNetwork(
            vector_dim=config.state_dim,
            grid_size=180,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        self.target_critic = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # Noise process
        self.noise = OUNoise(config.action_dim)
        
        # Optimized replay buffer
        self.memory = ReplayBuffer(config.buffer_size, self.device)
        
        # Initialize gradients scaler for mixed precision training
        self.scaler = torch.amp.GradScaler(str(self.device))
        
    def train(self, batch_size: int) -> Tuple[float, float]:
        """Optimized training process with efficient batch processing"""
        if len(self.memory) < batch_size:
            return 0.0, 0.0
            
        # Sample batch - returns preprocessed tensors
        (grid_inputs, vector_inputs, actions, rewards,
         next_grid_inputs, next_vector_inputs, dones) = self.memory.sample(batch_size)
        
        # Update Critic
        with torch.amp.autocast(str(self.device)):
            with torch.no_grad():
                next_actions = self.target_actor(next_grid_inputs, next_vector_inputs)
                target_q = self.target_critic(next_grid_inputs, next_vector_inputs, next_actions)
                target_q = rewards + (1 - dones) * self.config.gamma * target_q
            
            current_q = self.critic(grid_inputs, vector_inputs, actions)
            critic_loss = F.mse_loss(current_q, target_q)
        
        # Optimize critic with gradient scaling
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)
        
        # Update Actor
        with torch.amp.autocast(str(self.device)):
            actor_actions = self.actor(grid_inputs, vector_inputs)
            actor_loss = -self.critic(grid_inputs, vector_inputs, actor_actions).mean()
        
        # Optimize actor with gradient scaling
        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)
        
        self.scaler.update()
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return actor_loss.item(), critic_loss.item()
        
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Efficient soft update implementation"""
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.config.tau) + 
                    param.data * self.config.tau
                )
                
    def store_transition(self, state: VehicleState, action: VehicleAction, 
                        reward: float, next_state: VehicleState, done: bool):
        """Store transition in optimized buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def select_action(self, state: VehicleState, add_noise: bool = True) -> VehicleAction:
        grid_input, vector_input = state.get_network_inputs()
        
        # Add batch dimension
        grid_input = grid_input.unsqueeze(0).to(self.device)
        vector_input = vector_input.unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action_tensor = self.actor(grid_input, vector_input)
            steering, acceleration = action_tensor[0].cpu().numpy()
        self.actor.train()
        
        if add_noise:
            noise = self.noise.sample()
            steering = np.clip(steering + noise[0], *self.config.action_bounds[0])
            acceleration = np.clip(acceleration + noise[1], *self.config.action_bounds[1])
        
        return VehicleAction(steering=steering, acceleration=acceleration)
    
    def save(self, path: str):
        """Save model weights and training state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model weights and training state"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.config = checkpoint['config']