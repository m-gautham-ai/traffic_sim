import sys
import os
import torch
import numpy as np
from torch.distributions import Normal
from collections import defaultdict

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from laneless_env import LanelessEnv
num_vehicles = 3    
class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001):
        super(PolicyNetwork, self).__init__()
        self.gamma = gamma

        # Shared layers
        self.linear1 = torch.nn.Linear(state_dim, 128)
        self.linear2 = torch.nn.Linear(128, 128)

        # Layers for action means (acceleration, steering)
        self.mean_layer = torch.nn.Linear(128, action_dim)
        
        # Layer for action standard deviation
        # We use a single log_std for simplicity. It's often learned.
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Episode-specific memory
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        """
        Defines the forward pass of the network.
        Outputs the mean for the action distribution.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        
        # Tanh activation to keep the mean between -1 and 1
        mean = torch.tanh(self.mean_layer(x))
        return mean

    def get_action(self, state, deterministic=False):
        """
        Selects an action. If deterministic, it returns the mean. 
        Otherwise, it samples from the distribution.
        """
        mean = self(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum()
        return action, log_prob

    def select_action(self, state):
        """
        Selects a stochastic action for training and stores the log probability.
        """
        action, log_prob = self.get_action(state, deterministic=False)
        self.log_probs.append(log_prob)
        return action.detach().numpy(), log_prob

    def update(self, rewards_batch, log_probs_batch):
        """
        Performs the REINFORCE update on a batch of episodes.
        """
        policy_loss = []
        # Process each episode in the batch
        for rewards_episode, log_probs_episode in zip(rewards_batch, log_probs_batch):
            # 1. Calculate discounted rewards for the current episode
            discounted_rewards = []
            R = 0
            for r in reversed(rewards_episode):
                R = r + self.gamma * R
                discounted_rewards.insert(0, R)
            
            # 2. Normalize rewards and convert to tensor
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            # 3. Calculate loss for the episode
            # log_probs_episode is a list of tensors from the same episode
            episode_loss = [-log_prob * R for log_prob, R in zip(log_probs_episode, discounted_rewards)]
            policy_loss.extend(episode_loss)

        # Update network if there's any loss to backpropagate
        if not policy_loss:
            return

        self.optimizer.zero_grad()
        # Sum the losses from all episodes in the batch and perform backprop
        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()
        self.optimizer.step()

def evaluate_policy(policy, eval_episodes=10):
    """Runs policy for eval_episodes and returns average reward in a multi-agent env."""
    eval_env = LanelessEnv(render_mode=None, max_vehicles=num_vehicles)
    total_reward = 0
    for i in range(eval_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            actions = {}
            active_agents = list(obs.keys())
            for agent_id in active_agents:
                # Use deterministic actions for evaluation
                action, _ = policy.get_action(obs[agent_id], deterministic=True)
                actions[agent_id] = action.detach().numpy()
            
            next_obs, rewards, terminated, truncated, _ = eval_env.step(actions)
            
            # Sum rewards for all agents at this step
            episode_reward += sum(rewards.values())
            
            obs = next_obs
            
            # The episode is done if the observation dictionary is empty
            if not obs:
                done = True
        total_reward += episode_reward
    
    avg_reward = total_reward / eval_episodes
    print(f"Evaluation over {eval_episodes} episodes: Average Reward = {avg_reward:.2f}")
    eval_env.close()
    return avg_reward

def train():
    # --- Main Training Script ---
    EVAL_FREQ = 25 # Evaluate every 25 episodes

    env = LanelessEnv(render_mode=None, max_vehicles=num_vehicles)
    obs, _ = env.reset()

    # Get state and action dimensions from the environment
    agent_id = list(obs.keys())[0]
    state_dim = env.observation_space[agent_id].shape[0]
    action_dim = env.action_space[agent_id].shape[0]

    policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)

    best_avg_reward = -np.inf

    # Per-agent buffers to store the current trajectory
    agent_rewards = defaultdict(list)
    agent_log_probs = defaultdict(list)

    # Global buffers to collect completed episodes for batch updates
    batch_rewards = []
    batch_log_probs = []
    steps_since_update = 0
    total_steps = 0

    for step in range(1, 50000):
        # print(f"Step {step}")
        # --- Collect actions and step the environment ---
        actions = {}
        current_log_probs = {}
        active_agents = list(obs.keys())
        for agent_id in active_agents:
            action, log_prob = policy.select_action(obs[agent_id])
            actions[agent_id] = action
            agent_log_probs[agent_id].append(log_prob)

        next_obs, rewards_dict, terminated, truncated, _ = env.step(actions)
        
        num_active_agents = len(active_agents)
        steps_since_update += num_active_agents
        total_steps += num_active_agents

        for agent_id in active_agents:
            if agent_id in rewards_dict:
                agent_rewards[agent_id].append(rewards_dict[agent_id])

        # --- Check for finished episodes and add them to the batch ---
        done_agents = {agent_id for agent_id, is_done in terminated.items() if is_done}
        done_agents.update({agent_id for agent_id, is_done in truncated.items() if is_done})

        for agent_id in done_agents:
            if agent_rewards.get(agent_id):
                # Add the completed trajectory to the global batch
                batch_rewards.append(agent_rewards[agent_id])
                batch_log_probs.append(agent_log_probs[agent_id])
                # CRITICAL: Clear the per-agent buffer now that the episode is over
                del agent_rewards[agent_id]
                del agent_log_probs[agent_id]

        # --- Perform a policy update if enough steps have passed or the batch is large enough ---
        if (step % 3000 == 0 or len(batch_rewards) >= 10) and batch_rewards :
            print(f"Updating policy with {len(batch_rewards)} episodes and {len(batch_log_probs)} log probabilities.")
            policy.update(batch_rewards, batch_log_probs)
            # Clear the global batch buffers after the update
            batch_rewards.clear()
            batch_log_probs.clear()
            steps_since_update = 0

        # --- Evaluate the policy and save the best model ---
        if step % 10000 == 0: # Evaluate roughly every 10000 steps
            avg_reward = evaluate_policy(policy)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(policy.state_dict(), 'best_model.pth')
                print(f"New best model saved with avg reward: {avg_reward:.2f}")

        obs = next_obs

        if not obs:
            print(f"All agents finished at step {step}. Resetting environment.")
            policy.update(batch_rewards, batch_log_probs)
            obs, _ = env.reset()
            # Reset buffers for the new set of agents
            agent_rewards = defaultdict(list)
            agent_log_probs = defaultdict(list)
    env.close()

def test_agent():
    env = LanelessEnv(render_mode='human', max_vehicles=num_vehicles)
    obs, _ = env.reset()
    agent_id = list(obs.keys())[0]
    state_dim = env.observation_space[agent_id].shape[0]
    action_dim = env.action_space[agent_id].shape[0]
    policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
    policy.load_state_dict(torch.load('best_reinforce_policy.pth'))
    policy.eval()

    current_state = obs[agent_id]
    total_reward = 0
    num_steps = 0
    for step in range(1500):
        with torch.no_grad():
            # During testing, take the deterministic action
            action, _ = policy.get_action(current_state, deterministic=True)
            action = action.detach().numpy()
        
        next_state_dict, reward_dict, terminated, truncated, _ = env.step({agent_id: action})
        
        reward = reward_dict.get(agent_id, 0)
        total_reward += reward
        
        current_state = next_state_dict.get(agent_id)
        env.render()

        # Check termination status for the current agent
        is_terminated = terminated.get(agent_id, False)
        is_truncated = truncated.get(agent_id, False)
        
        if is_terminated or is_truncated or current_state is None:
            print("Episode finished.")
            break
        num_steps += 1            
    print(f"Test finished. Total reward: {total_reward:.2f}")
    print(f"Number of steps: {num_steps}")
    env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train or test the REINFORCE agent.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Set to "train" to train the agent, or "test" to evaluate a saved model.')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test_agent()
    else:
        print("Invalid mode. Please choose 'train' or 'test'.")

    