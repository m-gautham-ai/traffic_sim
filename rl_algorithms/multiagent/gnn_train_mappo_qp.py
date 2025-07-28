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
import qpsolvers
from qpsolvers import solve_qp

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
        # Shared GNN layers for the actor
        self.actor_conv1 = GCNConv(node_feature_dim, 128)
        self.actor_conv2 = GCNConv(128, 128)
        self.actor_mean_head = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Head (Centralized)
        self.critic_conv1 = GCNConv(node_feature_dim, 128)
        self.critic_conv2 = GCNConv(128, 128)
        self.critic_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def actor_parameters(self):
        return list(self.actor_conv1.parameters()) + \
               list(self.actor_conv2.parameters()) + \
               list(self.actor_mean_head.parameters()) + [self.actor_log_std]

    def critic_parameters(self):
        return list(self.critic_conv1.parameters()) + \
               list(self.critic_conv2.parameters()) + \
               list(self.critic_fc.parameters())

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # --- Actor (Decentralized) ---
        actor_x = F.relu(self.actor_conv1(x, edge_index))
        actor_x = F.relu(self.actor_conv2(actor_x, edge_index))
        mean = torch.tanh(self.actor_mean_head(actor_x))
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        # --- Critic (Centralized) ---
        with torch.no_grad(): # Critic does not influence actor gradients
            critic_x = F.relu(self.critic_conv1(x, edge_index))
            critic_x = F.relu(self.critic_conv2(critic_x, edge_index))
        global_x = global_mean_pool(critic_x, batch)
        value = self.critic_fc(global_x)

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
            action_tensor = dist.mean
            
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

def get_flat_params_from(params):
    return torch.cat([p.data.view(-1) for p in params])

def set_flat_params_to(params, flat_params):
    prev_ind = 0
    for param in params:
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def train_mappo_qp():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    writer = SummaryWriter(f"runs/MAPPO_QP_{int(time.time())}")
    env = LanelessEnv(render_mode=None, max_vehicles=MAX_VEHICLES_TRAIN)
    policy = ActorCriticGNN_MAPPO(node_feature_dim=1, action_dim=1).to(device)
    critic_optimizer = optim.Adam(policy.critic_parameters(), lr=POLICY_LR)
    buffer = RolloutBuffer()

    total_timesteps = 100000
    EVAL_INTERVAL = 5 # Evaluate less frequently for this complex update
    eval_count = 0
    update_count = 0
    best_avg_reward = -float('inf')
    MAX_KL = 0.01 # Trust region size
    DAMPING = 0.1 # Damping for Hessian approximation

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
            log_prob = dist.log_prob(action_tensor).sum(dim=-1)

        actions_dict = {node_to_vehicle_map[i]: action_tensor[i].cpu().numpy() for i in range(action_tensor.size(0))}
        next_obs, rewards_dict, _, _, is_truncated = env.step(actions_dict)

        rewards_list = [rewards_dict.get(vid, 0.0) for vid in node_to_vehicle_map.values()]
        # Ensure rewards are scalar floats, not numpy arrays
        rewards_scalar = [r.item() if hasattr(r, 'item') else r for r in rewards_list]
        rewards_tensor = torch.tensor(rewards_scalar, dtype=torch.float32).unsqueeze(1).to(device)
        dones_tensor = torch.tensor([is_truncated] * len(node_to_vehicle_map), dtype=torch.float32).unsqueeze(1).to(device)
        
        buffer.add(graph_data, action_tensor, log_prob, rewards_tensor, value, dones_tensor[0])

        obs = next_obs

        if len(buffer.graphs) >= BATCH_SIZE:
            update_count += 1
            print(f"--- Step {timestep}: Starting MAPPO-QP Update #{update_count} ---")

            with torch.no_grad():
                last_graph, _ = env.get_graph(next_obs)
                if last_graph is not None and last_graph.num_nodes > 0:
                    _, last_value = policy(last_graph.to(device))
                else:
                    last_value = torch.tensor([[0.0]]).to(device)
            
            buffer.calculate_advantages(last_value, torch.tensor(is_truncated, dtype=torch.float32).to(device))

            # --- Prepare Update Data ---
            batched_graph = Batch.from_data_list(buffer.graphs).to(device)
            old_actions = torch.cat(buffer.actions).to(device)
            old_log_probs = torch.cat([p.flatten() for p in buffer.log_probs]).detach()
            advantages = torch.cat(buffer.advantages).detach()
            critic_returns = torch.stack(buffer.critic_returns).detach()

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- Actor Update via QP ---
            actor_params = policy.actor_parameters()
            old_actor_params = get_flat_params_from(actor_params)

            # 1. Calculate policy gradient g
            dist, _ = policy(batched_graph)
            log_probs = dist.log_prob(old_actions).sum(dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)
            actor_loss = -(ratio * advantages).mean()
            
            grad_g = torch.autograd.grad(actor_loss, actor_params)
            g = torch.cat([grad.view(-1) for grad in grad_g]).detach()

            # 2. Formulate and solve QP
            P = torch.eye(g.shape[0]) * DAMPING
            q = g.cpu().numpy()
            
            try:
                step_direction = -torch.from_numpy(solve_qp(P.numpy(), q, solver='osqp')).float()
                if torch.isnan(step_direction).any():
                     raise Exception("QP solver returned NaN.")

                # 3. Line search to find step size that satisfies KL constraint
                with torch.no_grad():
                    shs = 0.5 * (step_direction * (P @ step_direction)).sum()
                    step_size = torch.sqrt(MAX_KL / shs) if shs > 0 else torch.tensor(1.0)

                    for i in range(10): # Backtracking line search
                        new_params = old_actor_params + step_size * step_direction
                        set_flat_params_to(actor_params, new_params)

                        # Check KL divergence
                        new_dist, _ = policy(batched_graph)
                        set_flat_params_to(actor_params, old_actor_params) # Revert for old_dist
                        old_dist, _ = policy(batched_graph)
                        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

                        if kl <= MAX_KL:
                            set_flat_params_to(actor_params, new_params) # Accept step
                            break
                        step_size *= 0.5
                    else:
                        set_flat_params_to(actor_params, old_actor_params) # No improvement

            except Exception as e:
                print(f"QP Actor Update failed: {e}. Skipping actor update.")
                set_flat_params_to(actor_params, old_actor_params) # Ensure old params are restored

            # --- Critic Update ---
            for _ in range(EPOCHS_PER_UPDATE):
                _, new_values = policy(batched_graph)
                critic_loss = F.mse_loss(new_values, critic_returns.squeeze(2))
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
            
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
                    torch.save(policy.state_dict(), 'mappo_gnn_qp_best.pth')
                    print(f"New best MAPPO-QP model saved with avg reward {avg_reward}")

    writer.close()
    env.close()

if __name__ == '__main__':
    train_mappo_qp()
