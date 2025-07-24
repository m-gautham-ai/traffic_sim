import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
import numpy as np
from collections import defaultdict

from laneless_env import LanelessEnv
from gnn_policy import ActorCriticGNN

# --- PPO Hyperparameters ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
POLICY_LR = 3e-4
VALUE_LR = 1e-3
BATCH_SIZE = 2048 # Number of steps to collect before updating
EPOCHS_PER_UPDATE = 10

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, graph, action, log_prob, reward, value, done, agent_index):
        self.graphs.append(graph)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.agent_indices.append(agent_index)

    def calculate_advantages(self, last_value, done):
        advantages = torch.zeros(len(self.rewards), 1)
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t+1]
                next_value = self.values[t+1]
            
            delta = self.rewards[t] + GAMMA * next_value * next_non_terminal - self.values[t]
            gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
            advantages[t] = gae
        
        self.returns = advantages + torch.cat(self.values)
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def get_batch(self):
        return (
            self.graphs,
            torch.cat(self.actions),
            torch.cat(self.log_probs),
            self.advantages,
            self.returns,
            self.agent_indices
        )

    def clear(self):
        self.graphs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.agent_indices = []

# --- Evaluation Function ---
def evaluate_policy(policy, eval_episodes=10):
    max_vehicles = 10
    eval_env = LanelessEnv(render_mode=None, max_vehicles=max_vehicles)
    total_rewards = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            graph_data = eval_env.get_graph()
            if not graph_data or graph_data.num_nodes == 0:
                break
            
            with torch.no_grad():
                dist, _ = policy(graph_data)
                actions_tensor = dist.mean # Use deterministic actions for evaluation
            
            actions_dict = {vid: act.detach().cpu().numpy() for vid, act in zip(graph_data.vehicle_ids, actions_tensor)}
            
            next_obs, rewards, _, truncated, _ = eval_env.step(actions_dict)
            episode_reward += sum(rewards.values())
            
            done = not next_obs or any(truncated.values())
            obs = next_obs
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def train_ppo():
    # --- Initialization ---
    max_vehicles = 10
    total_train_steps = 500000
    eval_freq = 5 # Evaluate every 5 updates
    env = LanelessEnv(render_mode=None, max_vehicles=max_vehicles)
    policy = ActorCriticGNN(node_feature_dim=1, action_dim=2)
    optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)
    buffer = RolloutBuffer()

    obs, _ = env.reset()
    best_avg_reward = -np.inf
    updates = 0
    
    for step in range(1, total_train_steps + 1):
        graph_data = env.get_graph()
        if not graph_data or graph_data.num_nodes == 0:
            obs, _ = env.reset()
            continue

        # --- Collect Experience ---
        with torch.no_grad():
            dist, value = policy(graph_data)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        
        actions_dict = {vid: act.cpu().numpy() for vid, act in zip(graph_data.vehicle_ids, action)}
        next_obs, rewards_dict, _, truncated, _ = env.step(actions_dict)

        # Store experience for each agent
        for i, agent_id in enumerate(graph_data.vehicle_ids):
            reward = rewards_dict.get(agent_id, 0.0)
            done = not next_obs or truncated.get(agent_id, False)
            buffer.add(graph_data, action[i].unsqueeze(0), log_prob[i].unsqueeze(0), reward, value[i], done, i)

        obs = next_obs
        if not obs or any(truncated.values()):
            obs, _ = env.reset()

        # --- PPO Update ---
        if len(buffer.rewards) >= BATCH_SIZE:
            with torch.no_grad():
                # Get value of last state for advantage calculation
                last_graph = env.get_graph()
                if last_graph and last_graph.num_nodes > 0:
                    _, last_value = policy(last_graph)
                    last_value = last_value.mean() # Average value across all agents
                else:
                    last_value = torch.tensor(0.0)

            buffer.calculate_advantages(last_value, any(truncated.values()))
            graphs, old_actions, old_log_probs, advantages, returns, agent_indices = buffer.get_batch()

            for _ in range(EPOCHS_PER_UPDATE):
                # Re-evaluate each step from the buffer to get current policy values
                new_log_probs_list, new_values_list, entropy_list = [], [], []
                for i in range(len(graphs)):
                    dist, value_all_agents = policy(graphs[i])
                    
                    # Select the specific agent's value and action distribution
                    agent_idx = agent_indices[i]
                    agent_action = old_actions[i]
                    
                    new_log_prob = dist.log_prob(agent_action).sum(axis=-1, keepdim=True)[agent_idx]
                    agent_value = value_all_agents[agent_idx]
                    
                    new_log_probs_list.append(new_log_prob)
                    new_values_list.append(agent_value)
                    entropy_list.append(dist.entropy().mean()) # Use mean entropy over all agents in graph

                new_log_probs = torch.stack(new_log_probs_list)
                new_values = torch.stack(new_values_list)
                entropy = torch.stack(entropy_list).mean()

                # PPO Loss Calculation
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns - new_values).pow(2).mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            buffer.clear()
            updates += 1
            print(f"Step {step}: Policy Updated ({updates}).")

            # --- Evaluate Policy ---
            if updates % eval_freq == 0:
                avg_reward = evaluate_policy(policy)
                print(f"Step: {step}, Updates: {updates}, Avg. Reward: {avg_reward:.2f}")
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(policy.state_dict(), 'ppo_gnn_best.pth')
                    print(f"New best PPO model saved with avg reward {best_avg_reward:.2f}")

if __name__ == '__main__':
    train_ppo()
