import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

from laneless_env import LanelessEnv

# --- Constants ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
POLICY_LR = 3e-4
BATCH_SIZE = 2048
EPOCHS_PER_UPDATE = 10
MAX_VEHICLES_EVAL = 10
MAX_VEHICLES_TRAIN = 10
MAX_EVAL_STEPS = 10000

# --- MAPPO Actor-Critic with GNN ---
class ActorCriticGNN_MAPPO(nn.Module):
    def __init__(self, node_feature_dim, action_dim):
        super(ActorCriticGNN_MAPPO, self).__init__()
        # Shared GNN layers
        self.conv1 = GCNConv(node_feature_dim, 128)
        self.conv2 = GCNConv(128, 128)

        # Actor Head (Decentralized)
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Head (Centralized)
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Shared GNN processing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # --- Actor (Decentralized) ---
        mean = torch.tanh(self.actor_mean(x)) # Action mean for each node
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        # --- Critic (Centralized) ---
        # Aggregate node features for a global state representation
        global_x = global_mean_pool(x, batch)
        value = self.critic_fc(global_x) # Single value for the entire graph (state)

        return dist, value


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, graph, actions, log_probs, rewards, value, dones):
        self.graphs.append(graph)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(value)
        self.dones.append(dones)

    def calculate_advantages(self, last_value, last_done):
        # For actor (decentralized advantages)
        last_gae_lam = torch.zeros(self.rewards[-1].shape[0], 1).to(last_value.device)
        # For critic (centralized returns)
        last_central_gae_lam = 0

        self.advantages = []
        self.critic_returns = []

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # --- Decentralized Advantage Calculation (for Actor) ---
            rewards = self.rewards[t]
            num_agents_t = rewards.shape[0]
            current_value_broadcast = self.values[t].expand(num_agents_t, 1)
            next_value_broadcast = next_value.expand(num_agents_t, 1)

            if last_gae_lam.shape[0] != num_agents_t:
                last_gae_lam_padded = torch.zeros(num_agents_t, 1).to(last_gae_lam.device)
                slice_len = min(num_agents_t, last_gae_lam.shape[0])
                last_gae_lam_padded[:slice_len] = last_gae_lam[:slice_len]
                last_gae_lam = last_gae_lam_padded

            delta = rewards + GAMMA * next_value_broadcast * next_non_terminal - current_value_broadcast
            last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            self.advantages.insert(0, last_gae_lam)

            # --- Centralized Return Calculation (for Critic) ---
            mean_reward = rewards.mean()
            current_value = self.values[t]
            
            central_delta = mean_reward + GAMMA * next_value * next_non_terminal - current_value
            last_central_gae_lam = central_delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_central_gae_lam
            self.critic_returns.insert(0, last_central_gae_lam + current_value)

    def clear(self):
        self.graphs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.critic_returns = []

def evaluate_mappo(policy, device, eval_episodes=10, max_eval_steps=MAX_EVAL_STEPS):
    policy.eval() # Set the policy to evaluation mode
    env = LanelessEnv(render_mode=None, max_vehicles=MAX_VEHICLES_EVAL)
    total_steps = 0
    obs, _ = env.reset()
    
    final_rewards_dict = {}
    vehicle_ids_done_dict = {}
    total_crashes = 0
    total_successes = 0

    with torch.no_grad(): # Disable gradient calculations for evaluation
        while total_steps < max_eval_steps:
            graph_data, node_to_vehicle_map = env.get_graph(obs)

            while graph_data is None or graph_data.num_nodes == 0:
                next_obs, _, _, _, _ = env.step({})
                obs = next_obs
                graph_data, node_to_vehicle_map = env.get_graph(obs)
                total_steps += 1
                if total_steps >= max_eval_steps: break
            if total_steps >= max_eval_steps: break

            dist, _ = policy(graph_data.to(device))
            action_tensor = dist.sample()
            
            actions_dict = {node_to_vehicle_map[i]: action_tensor[i].item() for i in range(action_tensor.size(0))}
            next_obs, rewards_dict, terminated_vehicles, off_screen_vehicles, is_truncated = env.step(actions_dict)

            for vid in terminated_vehicles:
                if vid not in vehicle_ids_done_dict:
                    total_crashes += 1
                    final_rewards_dict[vid] = rewards_dict.get(vid, -1)
                    vehicle_ids_done_dict[vid] = True

            for vid in off_screen_vehicles:
                if vid not in vehicle_ids_done_dict:
                    total_successes += 1
                    final_rewards_dict[vid] = rewards_dict.get(vid, 1)
                    vehicle_ids_done_dict[vid] = True

            obs = next_obs
            total_steps += 1

    policy.train() # Set the policy back to training mode
    avg_reward = sum(final_rewards_dict.values()) / len(final_rewards_dict) if final_rewards_dict else 0
    return avg_reward, total_crashes, total_successes

def train_mappo():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    writer = SummaryWriter(f"runs/MAPPO_{int(time.time())}")
    env = LanelessEnv(render_mode=None, max_vehicles=MAX_VEHICLES_TRAIN)
    policy = ActorCriticGNN_MAPPO(node_feature_dim=1, action_dim=1).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)
    buffer = RolloutBuffer()

    total_timesteps = 100000
    EVAL_INTERVAL = 2
    eval_count = 0
    update_count = 0
    best_avg_reward = -float('inf')

    obs, _ = env.reset()

    for timestep in range(1, total_timesteps + 1):
        # --- Collect Experience ---
        graph_data, node_to_vehicle_map = env.get_graph(obs)

        if graph_data is None or graph_data.num_nodes == 0:
            next_obs, _, _, _, _ = env.step({})
            obs = next_obs
            continue

        with torch.no_grad():
            dist, value = policy(graph_data.to(device))
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor).sum()

        actions_dict = {node_to_vehicle_map[i]: action_tensor[i].item() for i in range(action_tensor.size(0))}
        next_obs, rewards_dict, _, _, is_truncated = env.step(actions_dict)

        rewards_tensor = torch.tensor([rewards_dict.get(vid, 0.0) for vid in node_to_vehicle_map.values()], dtype=torch.float32).unsqueeze(1).to(device)
        dones_tensor = torch.tensor([is_truncated] * len(node_to_vehicle_map), dtype=torch.float32).unsqueeze(1).to(device)
        
        buffer.add(graph_data, action_tensor, log_prob, rewards_tensor, value, dones_tensor[0]) # Only need one done flag for the whole state

        obs = next_obs

        if len(buffer.graphs) >= BATCH_SIZE:
            update_count += 1
            print(f"--- Step {timestep}: Starting MAPPO Update #{update_count} ---")

            with torch.no_grad():
                last_graph, _ = env.get_graph(next_obs)
                if last_graph is not None and last_graph.num_nodes > 0:
                    _, last_value = policy(last_graph.to(device))
                else:
                    last_value = torch.tensor([[0.0]]).to(device)
            
            buffer.calculate_advantages(last_value, torch.tensor(is_truncated, dtype=torch.float32).to(device))

            # --- PPO Update ---
            batched_graph = Batch.from_data_list(buffer.graphs).to(device)
            old_actions = torch.cat(buffer.actions).to(device)
            old_log_probs = torch.stack(buffer.log_probs).to(device)
            advantages = torch.cat(buffer.advantages).detach()
            # For the critic, we need to reshape the returns to match the output shape
            critic_returns = torch.stack(buffer.critic_returns).detach()

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(EPOCHS_PER_UPDATE):
                dist, new_values = policy(batched_graph)
                new_log_probs = dist.log_prob(old_actions)

                # The critic loss is centralized
                # new_values is (batch_size, 1), critic_returns is (batch_size, 1, 1)
                # We squeeze the returns to match
                critic_loss = F.mse_loss(new_values, critic_returns.squeeze(2))

                # The actor loss is decentralized, but uses the centralized advantage
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy = dist.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            buffer.clear()
            print("--- Policy Updated ---")

            # --- Evaluate Policy ---
            if update_count % EVAL_INTERVAL == 0:
                eval_count += 1
                avg_reward, total_crashes, total_successes = evaluate_mappo(policy, device)
                print(f"Avg. Reward: {avg_reward}, Crashes: {total_crashes}, Successes: {total_successes}")
                writer.add_scalar('Evaluation/AverageReward', avg_reward, eval_count)
                writer.add_scalar('Evaluation/TotalCrashes', total_crashes, eval_count)
                writer.add_scalar('Evaluation/TotalSuccesses', total_successes, eval_count)
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(policy.state_dict(), 'mappo_gnn_best.pth')
                    print(f"New best MAPPO model saved with avg reward {avg_reward}")

    writer.close()
    env.close()

if __name__ == '__main__':
    train_mappo()

