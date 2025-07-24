import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
from collections import defaultdict
import numpy as np

from laneless_env import LanelessEnv

# --- GNN Policy Network ---
class GNNPolicy(nn.Module):
    def __init__(self, node_feature_dim, action_dim):
        super().__init__()
        # GAT layers learn to weigh neighbors' importance
        self.conv1 = GATv2Conv(node_feature_dim, 16, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(16 * 4, 32, heads=2, concat=True, edge_dim=2)

        # MLP head to produce action parameters from node embeddings
        self.action_mean_head = nn.Linear(32 * 2, action_dim)
        
        # Learnable standard deviation for the action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Pass through GAT layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Get the mean action for each node
        mean = torch.tanh(self.action_mean_head(x))
        return mean

    def get_action(self, graph_data, deterministic=False):
        mean_actions = self(graph_data)
        std = torch.exp(self.log_std)
        dist = Normal(mean_actions, std)

        actions = mean_actions if deterministic else dist.sample()
        log_probs = dist.log_prob(actions).sum(axis=-1)

        return actions, log_probs

    def update(self, batch_rewards, batch_log_probs):
        self.optimizer.zero_grad()
        policy_loss = []
        for i in range(len(batch_rewards)):
            rewards = batch_rewards[i]
            log_probs = batch_log_probs[i]

            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + 0.99 * R
                discounted_rewards.insert(0, R)
            
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
            # Normalize rewards only if there's more than one step to avoid std() of a single element being 0
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            for log_prob, reward in zip(log_probs, discounted_rewards):
                policy_loss.append(-log_prob * reward)

        if not policy_loss:
            return

        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()


# --- Main Training Script ---
def train_gnn():
    num_vehicles = 3
    env = LanelessEnv(render_mode=None, max_vehicles=num_vehicles)
    obs, _ = env.reset()

    state_dim = 1 # Just speed
    action_dim = 2 # acceleration, steering

    policy = GNNPolicy(node_feature_dim=state_dim, action_dim=action_dim)

    agent_rewards = defaultdict(list)
    agent_log_probs = defaultdict(list)

    batch_rewards = []
    batch_log_probs = []
    steps_since_update = 0

    for step in range(1, 50001):
        graph_data = env.get_graph()
        if not graph_data or graph_data.num_nodes == 0:
            obs, _ = env.reset()
            continue

        # --- Get actions for all agents in the graph ---
        all_actions_tensor, all_log_probs_tensor = policy.get_action(graph_data)
        actions_dict = {vid: act.detach().numpy() for vid, act in zip(graph_data.vehicle_ids, all_actions_tensor)}

        # --- Step the environment ---
        next_obs, rewards_dict, terminated, truncated, _ = env.step(actions_dict)
        steps_since_update += graph_data.num_nodes # Add number of active agents to step counter

        # --- Store rewards and log_probs ---
        for i, vehicle_id in enumerate(graph_data.vehicle_ids):
            if vehicle_id in rewards_dict:
                agent_rewards[vehicle_id].append(rewards_dict[vehicle_id])
                # Clone the log_prob to create a new tensor that is still attached to the graph
                agent_log_probs[vehicle_id].append(all_log_probs_tensor[i].clone())

        # --- Handle finished episodes ---
        done_agents = {agent_id for agent_id, is_done in terminated.items() if is_done}
        done_agents.update({agent_id for agent_id, is_done in truncated.items() if is_done})

        for agent_id in done_agents:
            if agent_id in agent_rewards:
                batch_rewards.append(agent_rewards.pop(agent_id))
                batch_log_probs.append(agent_log_probs.pop(agent_id))

        # --- Policy Update ---
        if (steps_since_update >= 2000 or len(batch_rewards) >= 10) and batch_rewards:
            print(f"Updating policy at step {step} with {len(batch_rewards)} episodes and {steps_since_update} steps.")
            policy.update(batch_rewards, batch_log_probs)
            batch_rewards.clear()
            batch_log_probs.clear()
            steps_since_update = 0

        obs = next_obs
        # Check if any agent was truncated, which implies a global time limit reset
        was_truncated = any(truncated.values())
        if not obs or was_truncated:
            if was_truncated:
                print(f"Global time limit reached at step {step}. Resetting environment.")
            else:
                print(f"All agents finished at step {step}. Resetting environment.")
            obs, _ = env.reset()

if __name__ == '__main__':
    train_gnn()
