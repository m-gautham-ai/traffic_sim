import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv


# --- Actor-Critic GNN for PPO ---
class ActorCriticGNN(nn.Module):
    def __init__(self, node_feature_dim, action_dim):
        super().__init__()
        # Shared GNN base layers
        self.conv1 = GATv2Conv(node_feature_dim, 16, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(16 * 4, 32, heads=2, concat=True, edge_dim=2)

        # Actor Head
        self.actor_head = nn.Linear(32 * 2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic Head
        self.critic_head = nn.Linear(32 * 2, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Pass through shared layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Get actor and critic values
        action_mean = torch.tanh(self.actor_head(x))
        value = self.critic_head(x)

        # Create action distribution
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        return dist, value

# --- GNN Policy Network ---
class GNNPolicy(nn.Module):
    def __init__(self, node_feature_dim, action_dim):
        super().__init__()
        self.conv1 = GATv2Conv(node_feature_dim, 16, heads=4, concat=True, edge_dim=2)
        self.conv2 = GATv2Conv(16 * 4, 32, heads=2, concat=True, edge_dim=2)
        self.action_mean_head = nn.Linear(32 * 2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        mean = torch.tanh(self.action_mean_head(x))
        return mean

    def get_action(self, graph_data, deterministic=False):
        mean_actions = self(graph_data)
        std = torch.exp(self.log_std)
        dist = Normal(mean_actions, std)
        actions = mean_actions if deterministic else dist.sample()
        log_probs = dist.log_prob(actions).sum(axis=-1)
        return actions, log_probs

    def evaluate_actions(self, graph_data, action_taken, agent_index):
        # Get action means for all nodes in the graph
        mean_actions = self(graph_data)
        # Select the mean action for the specific agent we're evaluating
        agent_mean_action = mean_actions[agent_index]
        
        std = torch.exp(self.log_std)
        dist = Normal(agent_mean_action, std)
        log_prob = dist.log_prob(action_taken).sum() # Sum over the action dimensions
        return log_prob


