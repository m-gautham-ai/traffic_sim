import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.optim as optim
from collections import defaultdict
import numpy as np

from laneless_env import LanelessEnv
from gnn_policy import GNNPolicy

max_vehicles = 10
total_train_steps = 500000
# --- Policy Update Function ---
def update_policy(policy, optimizer, trajectory_batch):
    optimizer.zero_grad()
    policy_loss = []

    for trajectory in trajectory_batch:
        rewards = trajectory['rewards']
        graph_list = trajectory['graphs']
        action_list = trajectory['actions']
        agent_indices = trajectory['agent_indices']

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Evaluate each step
        for i in range(len(graph_list)):
            log_prob = policy.evaluate_actions(graph_list[i], action_list[i], agent_indices[i])
            policy_loss.append(-log_prob * discounted_rewards[i])

    if not policy_loss:
        return

    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()

# --- Evaluation Function ---
def evaluate_policy(policy, eval_episodes=10):
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
            
            actions_tensor, _ = policy.get_action(graph_data, deterministic=True)
            actions_dict = {vid: act.detach().numpy() for vid, act in zip(graph_data.vehicle_ids, actions_tensor)}
            
            next_obs, rewards, terminated, truncated, _ = eval_env.step(actions_dict)
            episode_reward += sum(rewards.values())
            
            done = not next_obs or any(truncated.values())
            obs = next_obs
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)



# --- Main Training Script ---
def train_gnn_stable():
    EVAL_FREQ = 25 # Evaluate every 25 policy updates
    # num_vehicles = 3
    env = LanelessEnv(render_mode=None, max_vehicles=max_vehicles)
    obs, _ = env.reset()

    policy = GNNPolicy(node_feature_dim=1, action_dim=2)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    agent_trajectories = defaultdict(lambda: {'graphs': [], 'actions': [], 'rewards': [], 'agent_indices': []})
    batch_trajectories = []
    steps_since_update = 0
    best_avg_reward = -np.inf
    updates = 0

    for step in range(1,total_train_steps):
        graph_data = env.get_graph()
        if not graph_data or graph_data.num_nodes == 0:
            obs, _ = env.reset()
            continue

        actions_tensor, _ = policy.get_action(graph_data)
        actions_dict = {vid: act.detach().numpy() for vid, act in zip(graph_data.vehicle_ids, actions_tensor)}

        next_obs, rewards_dict, terminated, truncated, _ = env.step(actions_dict)
        steps_since_update += graph_data.num_nodes

        for i, agent_id in enumerate(graph_data.vehicle_ids):
            reward = rewards_dict.get(agent_id, 0.0)
            agent_trajectories[agent_id]['graphs'].append(graph_data)
            agent_trajectories[agent_id]['actions'].append(actions_tensor[i])
            agent_trajectories[agent_id]['rewards'].append(reward)
            agent_trajectories[agent_id]['agent_indices'].append(i)

        done_agents = {agent_id for agent_id, is_done in terminated.items() if is_done} \
                      | {agent_id for agent_id, is_done in truncated.items() if is_done}

        for agent_id in done_agents:
            if agent_id in agent_trajectories:
                batch_trajectories.append(agent_trajectories.pop(agent_id))

        if (steps_since_update >= 4000 or len(batch_trajectories) >= 20) and batch_trajectories:
            update_policy(policy, optimizer, batch_trajectories)
            batch_trajectories.clear()
            steps_since_update = 0
            updates += 1

            if updates % EVAL_FREQ == 0:
                avg_reward = evaluate_policy(policy)
                print(f"Step: {step}, Avg. Reward: {avg_reward:.2f}")
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(policy.state_dict(), 'gnn_policy_best.pth')
                    print(f"New best model saved with avg. reward {best_avg_reward:.2f}")

        obs = next_obs
        was_truncated = any(truncated.values())
        if not obs or was_truncated:
            obs, _ = env.reset()

if __name__ == '__main__':
    train_gnn_stable()
