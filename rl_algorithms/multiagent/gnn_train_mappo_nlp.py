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
from scipy.optimize import minimize


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
        critic_x = F.relu(self.critic_conv1(x, edge_index))
        critic_x = F.relu(self.critic_conv2(critic_x, edge_index))
        # The critic produces a single value for the entire state (graph)
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
        last_gae_lam = 0
        self.advantages = [0] * len(self.rewards)
        self.critic_returns = [0] * len(self.rewards)

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t+1]
            
            # Centralized value function estimates the value of the whole state
            # So we use the mean reward for the critic's return calculation
            mean_reward = self.rewards[t].mean()
            delta = mean_reward + GAMMA * next_value * next_non_terminal - self.values[t]
            last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
            self.critic_returns[t] = last_gae_lam + self.values[t]

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
    env = LanelessEnv(render_mode=None, max_vehicles=MAX_VEHICLES_EVAL, evaluation_mode=True)
    total_steps = 0
    obs, _ = env.reset()
    
    final_rewards_dict = {}
    vehicle_ids_done_dict = {}
    total_crashes = 0
    total_successes = 0

    with torch.no_grad(): # Disable gradient calculations for evaluation
        while total_steps < max_eval_steps:
            graph_data, node_to_vehicle_map = env.get_graph(obs)

            if graph_data is None or graph_data.num_nodes == 0:
                # If no vehicles, step with empty action to advance simulation
                next_obs, _, terminated, off_screen, _ = env.step({})
                obs = next_obs
                total_steps += 1
                if not env.simulation.sprites: # End if all vehicles are gone
                    break
                continue

            dist, _ = policy(graph_data.to(device))
            # Use the mean of the distribution for deterministic evaluation actions
            action_tensor = dist.mean 
            
            actions_dict = {node_to_vehicle_map[i]: action_tensor[i].cpu().numpy() for i in range(action_tensor.size(0))}
            next_obs, rewards_dict, terminated_vehicles, off_screen_vehicles, is_truncated = env.step(actions_dict)

            # Record final rewards for vehicles that terminated or went off-screen
            for v in terminated_vehicles:
                if v.id not in vehicle_ids_done_dict:
                    total_crashes += 1
                    final_rewards_dict[v.id] = rewards_dict.get(v.id, -100) # Use the reward from the dict
                    vehicle_ids_done_dict[v.id] = True

            for v in off_screen_vehicles:
                if v.id not in vehicle_ids_done_dict:
                    total_successes += 1
                    final_rewards_dict[v.id] = rewards_dict.get(v.id, 200) # Use the reward from the dict
                    vehicle_ids_done_dict[v.id] = True

            obs = next_obs
            total_steps += 1
            
            # If all vehicles are gone, end the evaluation loop
            if not env.simulation.sprites:
                break

    policy.train() # Set the policy back to training mode
    avg_reward = sum(final_rewards_dict.values()) / len(final_rewards_dict) if final_rewards_dict else 0
    return avg_reward, total_crashes, total_successes


def train_mappo_nlp(total_timesteps=500000, entropy_coeff=0.01, value_loss_coeff=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(f"runs/MAPPO_NLP_PPO_{int(time.time())}")
    env = LanelessEnv(render_mode=None, max_vehicles=MAX_VEHICLES_TRAIN)
    
    policy = ActorCriticGNN_MAPPO(node_feature_dim=2, action_dim=1).to(device)
    actor_optimizer = optim.Adam(policy.actor_parameters(), lr=POLICY_LR)
    critic_optimizer = optim.Adam(policy.critic_parameters(), lr=POLICY_LR)
    
    buffer = RolloutBuffer()

    obs, _ = env.reset()
    global_step = 0
    update_step = 0

    while global_step < total_timesteps:
        # --- Collect Rollouts ---
        for _ in range(BATCH_SIZE):
            graph_data, node_to_vehicle_map = env.get_graph(obs)

            if graph_data is None or graph_data.num_nodes == 0:
                next_obs, _, _, _, is_truncated = env.step({})
                obs = next_obs
                global_step += 1
                if is_truncated or not env.simulation.sprites: obs, _ = env.reset()
                continue

            graph_data = graph_data.to(device)
            with torch.no_grad():
                dist, value = policy(graph_data)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            actions_dict = {node_to_vehicle_map[i]: action[i].cpu().numpy() for i in range(action.size(0))}
            next_obs, rewards_dict, terminated_vehicles, off_screen_vehicles, is_truncated = env.step(actions_dict)

            num_nodes = graph_data.num_nodes
            rewards_tensor = torch.zeros(num_nodes, 1, device=device)
            dones_tensor = torch.zeros(1, 1, device=device)
            if is_truncated or terminated_vehicles or off_screen_vehicles:
                 dones_tensor[0,0] = 1

            for i in range(num_nodes):
                v_id = node_to_vehicle_map[i]
                rewards_tensor[i] = float(rewards_dict.get(v_id, 0))

            buffer.add(graph_data.cpu(), action.cpu(), log_prob.cpu(), rewards_tensor.cpu(), value.cpu(), dones_tensor.cpu())
            obs = next_obs
            global_step += 1
            if is_truncated or not env.simulation.sprites:
                obs, _ = env.reset()

        # --- PPO Update Phase ---
        update_step += 1
        print(f"--- Step {global_step}: Starting PPO Update #{update_step} ---")

        with torch.no_grad():
            graph_data, _ = env.get_graph(obs)
            if graph_data is None or graph_data.num_nodes == 0:
                last_value = torch.zeros(1, 1)
            else:
                _, last_value = policy(graph_data.to(device))
            last_done = torch.zeros(1, 1)
        buffer.calculate_advantages(last_value.cpu(), last_done.cpu())

        batched_graph = Batch.from_data_list(buffer.graphs).to(device)
        old_log_probs = torch.cat(buffer.log_probs).to(device)
        actions_batch = torch.cat(buffer.actions).to(device)
        
        # Decentralized advantages for actor
        advantages_decentralized = []
        for i, graph in enumerate(buffer.graphs):
            adv_for_graph = buffer.advantages[i].expand(graph.num_nodes, 1)
            advantages_decentralized.append(adv_for_graph)
        advantages_decentralized = torch.cat(advantages_decentralized).to(device)
        advantages_decentralized = (advantages_decentralized - advantages_decentralized.mean()) / (advantages_decentralized.std() + 1e-8)

        # Centralized returns for critic
        critic_returns = torch.cat(buffer.critic_returns).to(device)

        for _ in range(EPOCHS_PER_UPDATE):
            dist, values = policy(batched_graph)
            new_log_probs = dist.log_prob(actions_batch)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages_decentralized
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages_decentralized
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values, critic_returns)
            entropy_bonus = dist.entropy().mean()
            loss = actor_loss + value_loss_coeff * critic_loss - entropy_coeff * entropy_bonus

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()

        writer.add_scalar("train/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("train/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("train/entropy", entropy_bonus.item(), global_step)

        buffer.clear()

        if update_step % 2 == 0:
            avg_reward, crashes, successes = evaluate_mappo(policy, device)
            print(f"Step {global_step}: Eval Avg Reward: {avg_reward:.2f}, Crashes: {crashes}, Successes: {successes}")
            writer.add_scalar("eval/avg_reward", avg_reward, global_step)
            writer.add_scalar("eval/crashes", crashes, global_step)
            writer.add_scalar("eval/successes", successes, global_step)
            torch.save(policy.state_dict(), f"mappo_gnn_nlp_ppo_latest.pth")

    env.close()
    writer.close()


    

if __name__ == '__main__':
    train_mappo_nlp()
