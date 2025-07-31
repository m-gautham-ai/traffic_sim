import sys
import numpy as np
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
import qpsolvers


def calculate_theil_index(rewards):
    """Calculates the Theil index for a list of rewards."""
    rewards = np.array(rewards, dtype=np.float64)
    n = len(rewards)
    if n < 2:
        return 0.0

    min_reward = np.min(rewards)
    if min_reward < 0:
        rewards += -min_reward

    mean_reward = np.mean(rewards)
    if mean_reward == 0:
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log((rewards / mean_reward) + 1e-9)
        term = (rewards / mean_reward) * log_term
        term[np.isnan(term)] = 0
        theil_index = np.sum(term) / n

    return theil_index if not np.isnan(theil_index) else 0.0

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
CBF_GAMMA = 0.5  # CBF parameter

class ActorCriticGNN_MAPPO(nn.Module):
    def __init__(self, node_feature_dim, action_dim):
        super(ActorCriticGNN_MAPPO, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, 128)
        self.conv2 = GCNConv(128, 128)
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.critic_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        mean = torch.tanh(self.actor_mean(x))
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        global_x = global_mean_pool(x, batch)
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
        # Initialize last_gae_lam as a tensor of the correct size for the last step
        last_gae_lam = torch.zeros_like(self.rewards[-1])
        self.advantages = []
        self.returns = []

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t+1]
                next_value = self.values[t+1]

            # Handle shape changes in value tensors
            num_agents_t = self.rewards[t].shape[0]
            if next_value.shape[0] != num_agents_t:
                # This can happen if the last agent departs. Assume its value is 0.
                # We need a value for each agent at time t.
                # A global critic value is used, so we can just use the first element.
                next_value_broadcast = next_value[0].expand(num_agents_t, 1)
            else:
                next_value_broadcast = next_value

            delta = self.rewards[t] + GAMMA * next_value_broadcast * next_non_terminal - self.values[t]

            # Handle shape changes in advantage tensor (last_gae_lam)
            if last_gae_lam.shape[0] != num_agents_t:
                # Pad with zeros for agents that don't exist at t+1
                padded_gae_lam = torch.zeros(num_agents_t, 1).to(last_gae_lam.device)
                slice_len = min(num_agents_t, last_gae_lam.shape[0])
                padded_gae_lam[:slice_len] = last_gae_lam[:slice_len]
                last_gae_lam = padded_gae_lam

            last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            self.advantages.insert(0, last_gae_lam)
            self.returns.insert(0, last_gae_lam + self.values[t])

    def clear(self):
        self.graphs, self.actions, self.log_probs, self.rewards, self.values, self.dones = [], [], [], [], [], []
        self.advantages, self.returns = [], []

def apply_cbf(env, nominal_actions):
    safe_actions = {}
    vehicle_ids = list(nominal_actions.keys())
    # Create a dictionary of active vehicle objects for efficient lookup
    vehicle_objects = {v.id: v for v in env.simulation.sprites}

    for i, vid in enumerate(vehicle_ids):
        u_nominal = np.array([nominal_actions[vid]])

        # Skip if the vehicle for this action no longer exists
        if vid not in vehicle_objects:
            safe_actions[vid] = u_nominal[0]
            continue

        vehicle_i = vehicle_objects[vid]
        constraints = []

        for j, other_vid in enumerate(vehicle_ids):
            if i == j: continue
            
            # Skip if the other vehicle in the pair no longer exists
            if other_vid not in vehicle_objects:
                continue

            vehicle_j = vehicle_objects[other_vid]

            p_i = np.array([vehicle_i.x, vehicle_i.y])
            p_j = np.array([vehicle_j.x, vehicle_j.y])
            # Assuming velocity is primarily horizontal for 'right' direction vehicles
            v_i = np.array([vehicle_i.speed, 0.0])
            v_j = np.array([vehicle_j.speed, 0.0])

            d = p_i - p_j
            d_norm = np.linalg.norm(d)
            # The safety distance is based on the widths of the two vehicles
            safety_distance = (vehicle_i.width + vehicle_j.width) * 0.5 * 1.5 # Using a 1.5x safety factor
            h = d_norm**2 - safety_distance**2

            L_f_h = 2 * np.dot(d, v_i - v_j)
            # Action is acceleration, assume it affects velocity directly.
            # The g(x) term depends on the vehicle's dynamics model.
            # Assuming action is applied to y-velocity component.
            g_i = np.array([0, 1])
            g_j = np.array([0, 1])
            L_g_h = 2 * np.dot(d, g_i - g_j)

            if abs(L_g_h) > 1e-6: # Check for non-zero to avoid trivial constraints
                constraints.append((np.array([[L_g_h]]), np.array([-L_f_h - CBF_GAMMA * h])))

        if not constraints:
            safe_actions[vid] = u_nominal[0]
            continue

        # Setup and solve the QP
        G = np.vstack([c[0] for c in constraints])
        h_constr = np.vstack([c[1] for c in constraints])
        P = np.eye(1) * 2 # Quadratic term: (u - u_nominal)^2
        q = -2 * u_nominal # Linear term

        try:
            # qpsolvers expects float64
            u_safe = qpsolvers.solve_qp(P.astype(np.float64), q.astype(np.float64), G=G.astype(np.float64), h=h_constr.astype(np.float64), solver='cvxopt')
            safe_actions[vid] = u_safe[0] if u_safe is not None else u_nominal[0]
        except Exception as e:
            # If solver fails, fallback to nominal action
            safe_actions[vid] = u_nominal[0]

    return safe_actions

def evaluate_mappo(policy, device, eval_episodes=10, max_eval_steps=MAX_EVAL_STEPS):
    policy.eval()
    env = LanelessEnv(render_mode=None, max_vehicles=MAX_VEHICLES_EVAL)
    total_steps = 0
    vehicle_ids_done_dict = {}
    final_rewards_dict = {}
    total_crashes = 0
    total_successes = 0
    total_violating_vehicles = set()
    start_time = time.time()

    obs, _ = env.reset()
    with torch.no_grad():
        while total_steps < max_eval_steps:
            graph_data, node_to_vehicle_map = env.get_graph(obs)

            while graph_data is None or graph_data.num_nodes == 0:
                next_obs, _, _, _, _ = env.step({})
                graph_data, node_to_vehicle_map = env.get_graph(next_obs)
                total_steps += 1
                if total_steps >= max_eval_steps: break
            if total_steps >= max_eval_steps: break

            dist, _ = policy(graph_data.to(device))
            action_tensor = dist.mean
            actions_dict = {node_to_vehicle_map[i]: action_tensor[i].item() for i in range(action_tensor.size(0))}
            
            safe_actions = apply_cbf(env, actions_dict)

            next_obs, rewards_dict, terminated_vehicles, off_screen_vehicles, is_truncated = env.step(safe_actions)

            total_violating_vehicles.update(env.get_safety_violations())

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

    end_time = time.time()
    eval_duration_minutes = (end_time - start_time) / 60.0

    policy.train()
    avg_reward = sum(final_rewards_dict.values()) / len(final_rewards_dict) if final_rewards_dict else 0
    total_vehicles_in_eval = len(final_rewards_dict) if final_rewards_dict else 1
    safety_violation_rate = (len(total_violating_vehicles) / total_vehicles_in_eval) * 100 if total_steps > 0 else 0
    fairness_index = calculate_theil_index(list(final_rewards_dict.values()))
    throughput = total_successes / eval_duration_minutes if eval_duration_minutes > 0 else 0

    denominator = total_crashes + total_successes
    collision_rate = (total_crashes / denominator) * 100 if denominator > 0 else 0

    return avg_reward, total_crashes, total_successes, safety_violation_rate, fairness_index, throughput, collision_rate

def train_mappo_cbf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(f"runs/MAPPO_CBF_{int(time.time())}")
    env = LanelessEnv(render_mode=None, max_vehicles=MAX_VEHICLES_TRAIN)
    policy = ActorCriticGNN_MAPPO(node_feature_dim=2, action_dim=1).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)
    buffer = RolloutBuffer()

    total_timesteps = 100000
    EVAL_INTERVAL = 2
    eval_count = 0
    update_count = 0
    best_avg_reward = -float('inf')

    obs, _ = env.reset()

    for timestep in range(1, total_timesteps + 1):
        graph_data, node_to_vehicle_map = env.get_graph(obs)

        if graph_data is None or graph_data.num_nodes == 0:
            next_obs, _, _, _, _ = env.step({})
            obs = next_obs
            continue

        with torch.no_grad():
            dist, value = policy(graph_data.to(device))
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)

        actions_dict = {node_to_vehicle_map[i]: action_tensor[i].item() for i in range(action_tensor.size(0))}
        safe_actions = apply_cbf(env, actions_dict)
        next_obs, rewards_dict, _, _, is_truncated = env.step(safe_actions)

        rewards_tensor = torch.tensor([rewards_dict.get(vid, 0.0) for vid in node_to_vehicle_map.values()], dtype=torch.float32).unsqueeze(1).to(device)
        dones_tensor = torch.tensor([is_truncated] * len(node_to_vehicle_map), dtype=torch.float32).unsqueeze(1).to(device)
        
        buffer.add(graph_data, action_tensor, log_prob, rewards_tensor, value, dones_tensor[0])

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

            batched_graph = Batch.from_data_list(buffer.graphs).to(device)
            old_actions = torch.cat(buffer.actions).to(device)
            old_log_probs = torch.cat(buffer.log_probs).to(device)
            advantages = torch.cat(buffer.advantages).detach()
            returns = torch.cat(buffer.returns).detach()

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(EPOCHS_PER_UPDATE):
                dist, new_values = policy(batched_graph)
                new_log_probs = dist.log_prob(old_actions).sum(dim=-1)

                # The critic is centralized (per-graph), but returns are per-agent.
                # We must expand the critic's values to match the returns.
                expanded_new_values = new_values.repeat_interleave(batched_graph.batch.bincount(), dim=0)
                critic_loss = F.mse_loss(expanded_new_values.squeeze(), returns.squeeze())

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

            if update_count % EVAL_INTERVAL == 0:
                eval_count += 1
                avg_reward, total_crashes, total_successes, safety_violation_rate, fairness_index, throughput, collision_rate = evaluate_mappo(policy, device, max_eval_steps=MAX_EVAL_STEPS)
                writer.add_scalar('Evaluation/AverageReward', avg_reward, eval_count)
                writer.add_scalar('Evaluation/CollisionCount', total_crashes, eval_count)
                writer.add_scalar('Evaluation/Throughput_vehicles_per_min', throughput, eval_count)
                writer.add_scalar('Evaluation/ViolatingVehiclesRate', safety_violation_rate, eval_count)
                writer.add_scalar('Evaluation/FairnessIndex_Theil', fairness_index, eval_count)
                writer.add_scalar('Evaluation/CollisionRate_percent', collision_rate, eval_count)

                print(f"\n--- Evaluation {eval_count} ---")
                print(f"Avg Reward: {avg_reward:.2f}")
                print(f"Throughput (vehicles/min): {throughput:.2f}")
                print(f"Collision Count: {total_crashes}")
                print(f"Collision Rate (%): {collision_rate:.2f}")
                print(f"Violating Vehicles Rate (%): {safety_violation_rate:.2f}")
                print(f"Fairness Index (Theil): {fairness_index:.2f}")
                print("---------------------\n")

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(policy.state_dict(), 'mappo_cbf_gnn_best.pth')
                    print(f"New best MAPPO+CBF model saved with avg reward {avg_reward:.2f}")

    writer.close()
    env.close()

if __name__ == '__main__':
    train_mappo_cbf()
