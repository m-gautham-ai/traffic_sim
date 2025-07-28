import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# --- Add project root to sys.path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from laneless_env import LanelessEnv

# --- Constants ---
# Hierarchical Learning Params
HIGH_LEVEL_INTERVAL = 10 # High-level policy makes a decision every N steps
NUM_TACTICS = 4 # Number of high-level actions (e.g., maintain, overtake, merge)

# PPO Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
POLICY_LR = 3e-4
BATCH_SIZE = 2048
EPOCHS_PER_UPDATE = 10

# Environment/Training Params
MAX_VEHICLES_EVAL = 10
MAX_VEHICLES_TRAIN = 10
MAX_EVAL_STEPS = 10000
TOTAL_TIMESTEPS = 1_000_000
EVAL_INTERVAL = 10 # In terms of number of policy updates

# --- Hierarchical Rollout Buffer ---
class HierarchicalRolloutBuffer:
    def __init__(self):
        self.graphs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.tactics = []
        self.group_states = []
        self.group_log_probs = []

    def clear(self):
        del self.graphs[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]
        del self.tactics[:]
        del self.group_states[:]
        del self.group_log_probs[:]

# --- GiGPO Hierarchical Actor-Critic --- 

class GroupPolicy(nn.Module):
    """High-Level Policy: Observes the whole scene and selects a group tactic."""
    def __init__(self, node_feature_dim, num_tactics):
        super(GroupPolicy, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, 128)
        self.fc1 = nn.Linear(128, 64)

        # High-level Actor Head
        self.actor_head = nn.Linear(64, num_tactics)

        # High-level Critic Head
        self.critic_head = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        group_state = global_mean_pool(x, batch)
        x = F.relu(self.fc1(group_state))
        
        # Get tactic probabilities (actor) and group value (critic)
        tactic_logits = self.actor_head(x)
        tactic_probs = F.softmax(tactic_logits, dim=-1)
        group_value = self.critic_head(x)

        return tactic_probs, group_value

class IndividualPolicy(nn.Module):
    """Low-Level Policy: Takes local observations + a tactic and outputs an action."""
    def __init__(self, node_feature_dim, action_dim, num_tactics):
        super(IndividualPolicy, self).__init__()
        # Embed the tactic into a vector
        self.tactic_embedding = nn.Embedding(num_tactics, 16)
        
        # GNN layers
        self.conv1 = GCNConv(node_feature_dim + 16, 128) # Input features + tactic embedding
        self.conv2 = GCNConv(128, 128)

        # Actor Head
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Head
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, data, tactic):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Get tactic embedding and broadcast it to all nodes in the graph
        tactic_emb = self.tactic_embedding(tactic).squeeze(1)

        # If we get a single tactic for multiple nodes (during rollout),
        # we need to repeat it for each node.
        # If we get a batch of tactics for a batch of nodes (during update),
        # the dimensions already match.
        if tactic_emb.dim() == 1 or tactic_emb.size(0) != x.size(0):
            tactic_emb = tactic_emb.repeat(x.size(0), 1)

        combined_features = torch.cat([x, tactic_emb], dim=1)

        # GNN processing
        x = F.relu(self.conv1(combined_features, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Actor output
        mean = torch.tanh(self.actor_mean(x))
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        # Critic output (centralized value)
        value = self.critic_fc(x) # Produce a value for each node (agent)

        return dist, value

def train_gigpo():
    """Main training loop for GiGPO."""
    print("Starting GiGPO Training...")
    writer = SummaryWriter(f"runs/GiGPO_{'your_run_name_here'}") # Customize your run name

    # --- Initialization ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    env = LanelessEnv(render_mode=None, max_vehicles=MAX_VEHICLES_TRAIN)
    obs, _ = env.reset()
    node_feature_dim = env.get_graph(obs)[0].num_node_features
    action_dim = 1 # Single continuous action for acceleration

    group_policy = GroupPolicy(node_feature_dim, NUM_TACTICS).to(device)
    individual_policy = IndividualPolicy(node_feature_dim, action_dim, NUM_TACTICS).to(device)

    group_optimizer = optim.Adam(group_policy.parameters(), lr=POLICY_LR)
    individual_optimizer = optim.Adam(individual_policy.parameters(), lr=POLICY_LR)

    buffer = HierarchicalRolloutBuffer()

    # --- Main Training Loop ---
    current_tactic = None
    tactic_log_prob = None
    group_state_data = None

    for t in range(1, TOTAL_TIMESTEPS + 1):
        # --- High-Level Action Selection ---
        if (t - 1) % HIGH_LEVEL_INTERVAL == 0:
            group_policy.eval()
            graph_data, _ = env.get_graph(obs)
            if graph_data and graph_data.num_nodes > 0:
                group_state_data = graph_data.to(device)
                tactic_probs, _ = group_policy(group_state_data)
                tactic_dist = torch.distributions.Categorical(tactic_probs)
                current_tactic = tactic_dist.sample()
                tactic_log_prob = tactic_dist.log_prob(current_tactic)
            group_policy.train()

        # Ensure we have a valid tactic before proceeding
        if current_tactic is None:
            obs, _, _, _, _ = env.step({})
            continue

        # --- Low-Level Action Selection & Experience Collection ---
        individual_policy.eval()
        graph_data, node_to_vehicle_map = env.get_graph(obs)
        if not graph_data or graph_data.num_nodes == 0:
            obs, _, _, _, _ = env.step({})
            continue
        
        graph_data = graph_data.to(device)
        dist, value = individual_policy(graph_data, current_tactic)
        action_tensor = dist.sample()
        log_prob_tensor = dist.log_prob(action_tensor).sum(dim=-1)
        individual_policy.train()

        actions_dict = {node_to_vehicle_map[i]: action_tensor[i].cpu().item() for i in range(action_tensor.size(0))}
        next_obs, rewards_dict, terminated_vehicles, off_screen_vehicles, is_truncated = env.step(actions_dict)

        # Store experience
        buffer.graphs.append(graph_data)
        buffer.actions.append(action_tensor)
        buffer.log_probs.append(log_prob_tensor)
        buffer.rewards.append(torch.tensor(list(rewards_dict.values()), dtype=torch.float32, device=device))
        buffer.dones.append(torch.ones(graph_data.num_nodes, device=device) if is_truncated else torch.zeros(graph_data.num_nodes, device=device))
        buffer.values.append(value)
        buffer.tactics.append(current_tactic)
        buffer.group_states.append(group_state_data)
        buffer.group_log_probs.append(tactic_log_prob)

        obs = next_obs

        # --- Policy Update ---
        if len(buffer.graphs) >= BATCH_SIZE:
            update_gigpo_policies(group_policy, individual_policy, group_optimizer, individual_optimizer, buffer, device, env, obs)
            buffer.clear()

    writer.close()
    env.close()

def update_gigpo_policies(group_policy, individual_policy, group_optimizer, individual_optimizer, buffer, device, env, last_obs):
    """Performs the PPO update for both high-level and low-level policies."""

    # --- 1. Calculate Low-Level Advantages (for Individual Policy) ---
    with torch.no_grad():
        # Get value of the last state to bootstrap GAE
        last_graph, _ = env.get_graph(last_obs)
        if not last_graph or last_graph.num_nodes == 0:
            last_value = torch.zeros(1, 1, device=device)
        else:
            last_tactic = buffer.tactics[-1]
            _, last_value = individual_policy(last_graph.to(device), last_tactic)
        if last_value.dim() == 1: last_value = last_value.unsqueeze(1)

        # Flatten the buffer to process experiences sequentially
        flat_rewards = torch.cat([r.flatten() for r in buffer.rewards])
        flat_values = torch.cat([v.flatten() for v in buffer.values])
        flat_dones = torch.cat([d.flatten() for d in buffer.dones])

        # DEBUG: Check shapes after flattening
        print(f"DEBUG: flat_rewards shape: {flat_rewards.shape}")
        print(f"DEBUG: flat_values shape: {flat_values.shape}")
        print(f"DEBUG: flat_dones shape: {flat_dones.shape}")

        # Ensure shapes are consistent after flattening
        assert flat_rewards.shape == flat_values.shape == flat_dones.shape, "Shape mismatch after flattening!"

        # Sequential GAE Calculation
        advantages = torch.zeros_like(flat_rewards)
        last_gae_lam = 0
        for t in reversed(range(len(flat_rewards))):
            if t == len(flat_rewards) - 1:
                next_non_terminal = 1.0
                next_value = last_value[0] # Use the bootstrapped value
            else:
                # If the next step is a real 'done', there is no next value
                next_non_terminal = 1.0 - flat_dones[t + 1].float()
                next_value = flat_values[t + 1]

            delta = flat_rewards[t] + GAMMA * next_value * next_non_terminal - flat_values[t]
            last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        low_level_returns = advantages + flat_values
        low_level_advantages = advantages

    # --- 2. Calculate High-Level Rewards and Advantages (for Group Policy) ---
    high_level_rewards = []
    high_level_values = []
    high_level_log_probs = []
    high_level_states = []

    # Aggregate rewards for each high-level step
    for i in range(0, len(buffer.rewards), HIGH_LEVEL_INTERVAL):
        chunk_rewards = buffer.rewards[i:i+HIGH_LEVEL_INTERVAL]
        # Sum of mean rewards over the interval
        aggregated_reward = sum([r.mean() for r in chunk_rewards])
        high_level_rewards.append(aggregated_reward)
        
        # Get the corresponding high-level state, value, and log_prob
        _, group_value = group_policy(buffer.group_states[i])
        high_level_values.append(group_value)
        high_level_log_probs.append(buffer.group_log_probs[i])
        high_level_states.append(buffer.group_states[i])

    # GAE for high-level policy
    with torch.no_grad():
        _, last_group_value = group_policy(buffer.group_states[-1])
        high_level_advantages = []
        high_level_returns = []
        last_gae_lam = 0
        for t in reversed(range(len(high_level_rewards))):
            if t == len(high_level_rewards) - 1:
                next_non_terminal = 1.0
                next_value = last_group_value
            else:
                next_non_terminal = 1.0 # Simplified, assumes episode doesn't end mid-batch
                next_value = high_level_values[t+1]
            
            rewards = high_level_rewards[t]
            values = high_level_values[t]
            delta = rewards + (GAMMA**HIGH_LEVEL_INTERVAL) * next_value * next_non_terminal - values
            last_gae_lam = delta + (GAMMA**HIGH_LEVEL_INTERVAL) * GAE_LAMBDA * next_non_terminal * last_gae_lam
            high_level_advantages.insert(0, last_gae_lam)
            high_level_returns.insert(0, last_gae_lam + values)

    # --- 3. Perform PPO Updates --- 
    # Note: This is a simplified update loop. A full implementation would use mini-batching.
    
    # --- Update Individual Policy ---
    # Prepare batched data for the GNN
    batched_graphs = Batch.from_data_list(buffer.graphs)
    b_actions = torch.cat(buffer.actions)
    b_log_probs_old = torch.cat([p.flatten() for p in buffer.log_probs]).detach()
    b_advantages = low_level_advantages
    b_returns = low_level_returns
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    # Expand tactics to match the number of nodes in each graph of the batch
    b_tactics = torch.cat([t.repeat(g.num_nodes, 1) for t, g in zip(buffer.tactics, buffer.graphs)])

    for _ in range(EPOCHS_PER_UPDATE):
        dist, values = individual_policy(batched_graphs, b_tactics)
        new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        # Policy loss
        ratio = (new_log_probs - b_log_probs_old).exp()
        surr1 = ratio * b_advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * b_advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        critic_loss = F.mse_loss(b_returns, values.squeeze())

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        individual_optimizer.zero_grad()
        loss.backward()
        individual_optimizer.step()

    # Update Group Policy
    old_group_log_probs = torch.cat(high_level_log_probs).detach()
    group_advs = torch.cat(high_level_advantages)
    group_advs = (group_advs - group_advs.mean()) / (group_advs.std() + 1e-8)
    group_returns = torch.cat(high_level_returns)

    for _ in range(EPOCHS_PER_UPDATE):
        new_tactic_probs, new_group_values = group_policy(Batch.from_data_list(high_level_states))
        dist = torch.distributions.Categorical(new_tactic_probs)
        new_group_log_probs = dist.log_prob(torch.cat(buffer.tactics[::HIGH_LEVEL_INTERVAL]))
        
        ratio = (new_group_log_probs - old_group_log_probs).exp()
        surr1 = ratio * group_advs
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * group_advs
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_group_values, group_returns)
        entropy = dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        group_optimizer.zero_grad()
        loss.backward()
        group_optimizer.step()


if __name__ == '__main__':
    train_gigpo()
