import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

from rl_algorithms.multiagent.nlp_impl import LaneFreeEnv

# --- MAPPO Actor-Critic Model (MLP) ---
class ActorCriticMLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticMLP, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist, value

# --- MAPPO Training Loop ---
def train_mappo_nlp_env():
    # Hyperparameters
    LR = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    EPOCHS = 10
    BATCH_SIZE = 2048
    MAX_TIMESTEPS = 100000

    env = LaneFreeEnv()
    device = torch.device("cpu")
    policy = ActorCriticMLP(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    writer = SummaryWriter(f"runs/MAPPO_NLP_ENV_{int(time.time())}")

    obs = env.reset()
    timestep = 0
    update_count = 0

    while timestep < MAX_TIMESTEPS:
        buffer_obs, buffer_actions, buffer_log_probs, buffer_rewards, buffer_dones, buffer_values = [], [], [], [], [], []
        
        for _ in range(BATCH_SIZE):
            timestep += 1
            obs_tensor = torch.from_numpy(obs).float().to(device)
            
            with torch.no_grad():
                dist, value = policy(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

            next_obs, reward, done, _ = env.step(action.cpu().numpy())

            buffer_obs.append(obs_tensor)
            buffer_actions.append(action)
            buffer_log_probs.append(log_prob)
            buffer_rewards.append(reward)
            buffer_dones.append(done)
            buffer_values.append(value.squeeze())

            obs = next_obs

        # GAE Calculation
        with torch.no_grad():
            next_obs_tensor = torch.from_numpy(next_obs).float().to(device)
            _, last_value = policy(next_obs_tensor)
            advantages = torch.zeros(BATCH_SIZE, device=device)
            gae = 0
            for t in reversed(range(len(buffer_rewards))):
                # If the episode was done, the value of the next state is 0
                next_val = buffer_values[t+1] if t < BATCH_SIZE - 1 else last_value
                next_non_terminal = 1.0 - buffer_dones[t]
                delta = buffer_rewards[t] + GAMMA * next_val * next_non_terminal - buffer_values[t]
                gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
                advantages[t] = gae
        
        returns = advantages + torch.tensor(buffer_values, device=device).clone().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        obs_batch = torch.stack(buffer_obs)
        actions_batch = torch.stack(buffer_actions)
        log_probs_batch = torch.stack(buffer_log_probs)

        for _ in range(EPOCHS):
            dist, values = policy(obs_batch)
            new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
            ratios = torch.exp(new_log_probs - log_probs_batch)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        update_count += 1
        avg_reward = np.mean(buffer_rewards)
        print(f"Timestep: {timestep}, Update #{update_count}, Avg Reward: {avg_reward:.2f}")
        writer.add_scalar('Training/AverageReward', avg_reward, timestep)
        writer.add_scalar('Loss/Total', loss.item(), timestep)

    writer.close()
    env.close()

if __name__ == '__main__':
    train_mappo_nlp_env()
