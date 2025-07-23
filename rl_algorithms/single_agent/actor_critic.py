import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    """
    Actor Network for Continuous Action Spaces.
    Outputs the mean and standard deviation of a Gaussian distribution.
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))
        mean = self.mean(x)
        
        # Clamp the log standard deviation for stability
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        return mean, std

class Critic(nn.Module):
    """
    Critic Network.
    Estimates the value of a state.
    """
    def __init__(self, state_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))
        return self.value(x)

class ActorCriticAgent:
    """
    A single-agent Actor-Critic implementation.
    """
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.max_action = max_action

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        print(state_tensor)
        mean, std = self.actor(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        # Clip the action to the valid range
        action = torch.clamp(action, -self.max_action, self.max_action)
            
        return action, action_log_prob

    def update(self, state, action, action_log_prob, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = action # It's already a tensor
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0)
        done_tensor = torch.FloatTensor([1 - done]).unsqueeze(0)

        # --- Update Critic ---
        # Calculate the target value
        target_value = reward_tensor + self.gamma * self.critic(next_state_tensor) * done_tensor
        
        # Calculate the current value and the advantage
        current_value = self.critic(state_tensor)
        advantage = (target_value - current_value).detach()
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_value, target_value.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Actor loss
        actor_loss = -(action_log_prob * advantage).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
