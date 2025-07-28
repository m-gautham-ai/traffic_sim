import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import time

from laneless_env import LanelessEnv
from gnn_policy import ActorCriticGNN
from torch_geometric.data import Batch

# --- PPO Hyperparameters ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
POLICY_LR = 3e-4
VALUE_LR = 1e-3
BATCH_SIZE = 2048 # Number of steps to collect before updating
MINI_BATCH_SIZE = 256 # Size of mini-batches for PPO update
EPOCHS_PER_UPDATE = 10

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, graph, actions, log_probs, rewards, values, dones):
        self.graphs.append(graph)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(dones)

    def calculate_advantages(self, last_value, last_done):
        # Ensure last_value is 2D
        if last_value.dim() == 1:
            last_value = last_value.unsqueeze(1)

        last_gae_lam = torch.zeros_like(last_value)
        self.advantages = []
        self.returns = []

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - torch.tensor(last_done, dtype=torch.float32).view(-1, 1)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].float()
                next_values = self.values[t + 1]

            rewards = self.rewards[t]
            values = self.values[t]

            # Ensure all tensors are 2D before comparison
            if values.dim() == 1: values = values.unsqueeze(1)
            if next_values.dim() == 1: next_values = next_values.unsqueeze(1)
            if rewards.dim() == 1: rewards = rewards.unsqueeze(1)
            if next_non_terminal.dim() == 1: next_non_terminal = next_non_terminal.unsqueeze(1)

            # Handle shape mismatch
            if values.shape[0] != next_values.shape[0]:
                num_agents_t = values.shape[0]
                slice_len = min(num_agents_t, next_values.shape[0])

                # Pad tensors to match the shape of the current timestep's values
                next_values_padded = torch.zeros_like(values)
                next_values_padded[:slice_len] = next_values[:slice_len]
                next_values = next_values_padded

                next_non_terminal_padded = torch.zeros_like(values)
                next_non_terminal_padded[:slice_len] = next_non_terminal[:slice_len]
                next_non_terminal = next_non_terminal_padded

                last_gae_lam_padded = torch.zeros_like(values)
                last_gae_lam_padded[:slice_len] = last_gae_lam[:slice_len]
                last_gae_lam = last_gae_lam_padded

            delta = rewards + GAMMA * next_values * next_non_terminal - values
            last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            self.advantages.insert(0, last_gae_lam)
            self.returns.insert(0, last_gae_lam + values)


    def get_batch(self, batch_size):
        n_timesteps = len(self.graphs)
        indices = np.random.permutation(n_timesteps)

        # Flatten all data across the time dimension for batching
        flat_graphs = self.graphs
        flat_actions = torch.cat(self.actions).detach()
        flat_log_probs = torch.cat(self.log_probs).detach()
        flat_advantages = torch.cat(self.advantages).detach()
        flat_returns = torch.cat(self.returns).detach()

        # Normalize advantages across the entire batch
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # We need to associate each agent-step with its original graph index
        graph_indices = torch.tensor([i for i, g in enumerate(flat_graphs) for _ in range(g.num_nodes)])

        n_samples = len(flat_actions)
        agent_indices = np.random.permutation(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_agent_indices = agent_indices[start_idx:end_idx]

            # Find the unique graphs these agents belong to
            unique_graph_indices = torch.unique(graph_indices[batch_agent_indices]).numpy()

            if len(unique_graph_indices) == 0:
                continue

            # Create a batch of only the necessary graphs
            graphs_to_batch = [flat_graphs[i] for i in unique_graph_indices]
            batched_graph = Batch.from_data_list(graphs_to_batch)

            # Now, we need a map from the original node index (in the full flattened data)
            # to the new node index in the batched graph.
            # This is the complex part we've been struggling with.
            # Let's simplify: we will batch by full graphs, not agent steps.

        # --- Simplified, Correct Batching by Timestep --- #
        for start_idx in range(0, n_timesteps, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            graphs_to_batch = [self.graphs[i] for i in batch_indices]
            actions_to_batch = [self.actions[i] for i in batch_indices]
            log_probs_to_batch = [self.log_probs[i] for i in batch_indices]
            advantages_to_batch = [self.advantages[i] for i in batch_indices]
            returns_to_batch = [self.returns[i] for i in batch_indices]

            batched_graph = Batch.from_data_list(graphs_to_batch)
            batch_actions = torch.cat(actions_to_batch)
            batch_log_probs = torch.cat(log_probs_to_batch)
            batch_advantages = torch.cat(advantages_to_batch)
            batch_returns = torch.cat(returns_to_batch)

            # Normalize advantages per mini-batch
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            yield batched_graph, batch_actions, batch_log_probs, batch_advantages, batch_returns

    def clear(self):
        self.graphs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

# --- Evaluation Function ---

def evaluate_ppo(policy,eval_episodes=10, max_eval_steps=1000) : 
  # --- Initialization ---
    max_vehicles = 15
    total_timesteps = 500000
    env = LanelessEnv(render_mode=None, max_vehicles=max_vehicles)

    total_steps = 0
    update_count = 0
    # Ensure the initial state is not empty
    obs, _ = env.reset()
    graph_data, _ = env.get_graph(obs)
    # while graph_data is None or graph_data.num_nodes == 0:
    #     print("Initial state is empty, resetting again...")
    #     # obs, _ = env.reset()
    #     graph_data, _ = env.get_graph()
    best_avg_reward = -float('inf')
    max_visible_vehicles = 0
    final_rewards_dict = {}
    vehicle_ids_done_dict = {}
    total_crashes = 0
    total_successes = 0
    while total_steps < max_eval_steps:
        # --- Collect Experience --- #
        graph_data, node_to_vehicle_map = env.get_graph(obs)

        while graph_data is None or graph_data.num_nodes == 0:
            print("No vehicles on screen. Stepping with no actions...")
            # total_steps += 1 # This is an environment step
            next_obs, _, _, _, _ = env.step({}) # Step with no actions
            graph_data, node_to_vehicle_map = env.get_graph(next_obs)

        dist, value = policy(graph_data)
        action_tensor = dist.sample()
        actions_dict = {node_to_vehicle_map[i]: action_tensor[i].item() for i in range(action_tensor.size(0))}
        next_obs, rewards_dict, terminated_vehicles, off_screen_vehicles, is_truncated = env.step(actions_dict)

        for vid, reward in rewards_dict.items():
            final_rewards_dict[vid] = final_rewards_dict.get(vid, 0.0) + reward

        for vid in terminated_vehicles:
            final_rewards_dict[vid] = final_rewards_dict.get(vid, 0.0) + -50.0
            vehicle_ids_done_dict[vid] = True

        for vid in off_screen_vehicles:
            final_rewards_dict[vid] = final_rewards_dict.get(vid, 0.0) + 10.0
            vehicle_ids_done_dict[vid] = True
        
        total_crashes += len(terminated_vehicles)
        total_successes += len(off_screen_vehicles)

        obs = next_obs
        total_steps += 1
    
    print ( "The number of vehicles that have finished the episode is: ", len(vehicle_ids_done_dict.keys()))
    
    my_reward = 0
    for vid in vehicle_ids_done_dict:
        my_reward += final_rewards_dict[vid]
    print ( "The total reward is: ", my_reward/len(vehicle_ids_done_dict.keys()))
    return my_reward/len(vehicle_ids_done_dict.keys()), total_crashes, total_successes

def train_ppo():
    # --- Initialization ---
    max_vehicles = 10
    total_timesteps = 100000
    EVAL_INTERVAL = 2 # Evaluate every 2 updates
    env = LanelessEnv(render_mode=None, max_vehicles=max_vehicles)
    policy = ActorCriticGNN(node_feature_dim=1, action_dim=1)
    optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)
    buffer = RolloutBuffer()
    eval_count = 0
    writer = SummaryWriter(f"runs/ppo_gnn_{int(time.time())}")

    total_steps = 0
    update_count = 0
    # Ensure the initial state is not empty
    obs, _ = env.reset()
    graph_data, _ = env.get_graph(obs)
    # while graph_data is None or graph_data.num_nodes == 0:
    #     print("Initial state is empty, resetting again...")
    #     # obs, _ = env.reset()
    #     graph_data, _ = env.get_graph()
    best_avg_reward = -float('inf')
    max_visible_vehicles = 0

    while total_steps < total_timesteps:
        # --- Collect Experience --- #
        total_crashes = 0
        total_successes = 0
        for _ in range(BATCH_SIZE):
            total_steps += 1
            graph_data, node_to_vehicle_map = env.get_graph(obs)

            # If no vehicles are present, step with no actions until they appear
            while graph_data is None or graph_data.num_nodes == 0:
                print("No vehicles on screen. Stepping with no actions...")
                total_steps += 1 # This is an environment step
                next_obs, _, _, _, _ = env.step({}) # Step with no actions
                graph_data, node_to_vehicle_map = env.get_graph(next_obs)

            with torch.no_grad():
                dist, value = policy(graph_data)
                action_tensor = dist.sample()
                log_prob_tensor = dist.log_prob(action_tensor)

            actions_dict = {node_to_vehicle_map[i]: action_tensor[i].item() for i in range(action_tensor.size(0))}
            next_obs, rewards_dict, terminated_vehicles, off_screen_vehicles, is_truncated = env.step(actions_dict)

            total_crashes += len(terminated_vehicles)
            total_successes += len(off_screen_vehicles)

            rewards_tensor = torch.tensor([rewards_dict.get(vid, 0.0) for vid in node_to_vehicle_map.values()], dtype=torch.float32)
            dones_tensor = torch.tensor([(vid in terminated_vehicles or is_truncated) for vid in node_to_vehicle_map.values()], dtype=torch.bool)

            vehicle_ids = list(node_to_vehicle_map.values())
            # print(f"Graph data shape: {graph_data}")
            # Track max visible vehicles
            if graph_data is not None and graph_data.num_nodes > max_visible_vehicles:
                max_visible_vehicles = graph_data.num_nodes
                print(f"New max visible vehicles: {max_visible_vehicles}")

            buffer.add(graph_data, action_tensor, log_prob_tensor, rewards_tensor, value, dones_tensor)

            obs = next_obs
            # if is_truncated:
            #     obs, _ = env.reset()
        print( len(buffer.graphs))
        # --- PPO Update --- #

        update_count += 1
        print(f"Crashes in batch: {total_crashes}")
        print(f"Successes in batch: {total_successes}")
        print(f"--- Step {total_steps}: Starting PPO Update #{update_count} ({len(buffer.graphs)} graphs) ---")
        print(f"Max visible vehicles: {max_visible_vehicles}")
        if graph_data is not None and graph_data.num_nodes > 0:
            # Flatten the speeds tensor and convert to a list for readability
            speeds = graph_data.x.view(-1).tolist()
            print(f"Vehicle Speeds: {[f'{s:.2f}' for s in speeds]}")

        print(f"Graph data: {graph_data}")
        print(f"Vehicle IDs: {graph_data.vehicle_ids}")
        with torch.no_grad():
            # The last observation is `next_obs` from the final step of the loop
            last_graph, _ = env.get_graph(next_obs)
            if last_graph is not None and last_graph.num_nodes > 0:
                _, last_value = policy(last_graph)
            else:
                # Provide a zero tensor with the correct shape if the last state is empty
                last_value = torch.zeros((0, 1))
        
        buffer.calculate_advantages(last_value, is_truncated)
        
        n_timesteps = len(buffer.graphs)
        if n_timesteps == 0: continue
        n_agents_per_graph = buffer.graphs[0].num_nodes
        graphs_per_minibatch = max(1, MINI_BATCH_SIZE // n_agents_per_graph)

        for _ in range(EPOCHS_PER_UPDATE):
            for batched_graph, batch_actions, batch_log_probs, batch_advantages, batch_returns in buffer.get_batch(graphs_per_minibatch):
                dist, new_values = policy(batched_graph)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - batch_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, batch_returns)

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        buffer.clear()
        print(f"--- Policy Updated ---")

        # --- Evaluate Policy ---
        if update_count > 0 and update_count % EVAL_INTERVAL == 0:
            eval_count += 1
            avg_reward, total_crashes, total_successes = evaluate_ppo(policy)
            print("Avg. Reward: ", avg_reward)
            writer.add_scalar('Evaluation/AverageReward', avg_reward, eval_count)
            writer.add_scalar('Evaluation/TotalCrashes', total_crashes, eval_count)
            writer.add_scalar('Evaluation/TotalSuccesses', total_successes, eval_count)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(policy.state_dict(), 'ppo_gnn_best.pth')
                print("New best PPO model saved with avg reward ", best_avg_reward)
    
    writer.close()

if __name__ == '__main__':
    train_ppo()
