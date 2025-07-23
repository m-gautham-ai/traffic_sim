import sys
import os
import torch
import numpy as np
from torch.distributions import Normal

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from laneless_env import LanelessEnv

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
        return action.detach().numpy()

    def update(self):
        """
        Performs the REINFORCE update.
        """
        discounted_rewards = []
        R = 0
        # Calculate discounted rewards
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        # Normalize rewards for stability
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * R)
        
        # Update network
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Clear memory for the next episode
        self.log_probs = []
        self.rewards = []

def evaluate_policy(policy, env, eval_episodes=10):
    """Runs policy for eval_episodes and returns average reward."""
    total_reward = 0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        agent_id = list(state.keys())[0]
        current_state = state[agent_id]
        episode_reward = 0
        
        for _ in range(1000): # Max steps per episode
            # During evaluation, take the deterministic action (the mean)
            action, _ = policy.get_action(current_state, deterministic=True)
            action = action.detach().numpy()
            next_state_dict, reward_dict, terminated, truncated, _ = env.step({agent_id: action})
            
            reward = reward_dict.get(agent_id, 0)
            episode_reward += reward
            
            current_state = next_state_dict.get(agent_id)

            # Check termination status for the current agent
            is_terminated = terminated.get(agent_id, False)
            is_truncated = truncated.get(agent_id, False)

            if is_terminated or is_truncated or current_state is None:
                break
        total_reward += episode_reward
    
    return total_reward / eval_episodes

def train():
    # --- Main Training Script ---
    EVAL_FREQ = 25 # Evaluate every 25 episodes

    env = LanelessEnv(render_mode=None, max_vehicles=1)
    obs, _ = env.reset()

    # Get state and action dimensions from the environment
    agent_id = list(obs.keys())[0]
    state_dim = env.observation_space[agent_id].shape[0]
    action_dim = env.action_space[agent_id].shape[0]

    policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)

    best_avg_reward = -np.inf

    for episode in range(1, 1001):
        state, info = env.reset()
        agent_id = list(state.keys())[0]
        current_state = state[agent_id]
        
        episode_reward = 0
        num_steps = 0
        for step in range(1000):
            action = policy.select_action(current_state)
            next_state_dict, reward_dict, terminated, truncated, info = env.step({agent_id: action})
            
            reward = reward_dict.get(agent_id, 0)
            policy.rewards.append(reward)
            episode_reward += reward
            
            current_state = next_state_dict.get(agent_id)

            # Check termination status for the current agent
            is_terminated = terminated.get(agent_id, False)
            is_truncated = truncated.get(agent_id, False)

            if is_terminated or is_truncated or current_state is None:
                break
            num_steps += 1                
        policy.update()

        print(f"Episode {episode} finished with reward {episode_reward:.2f} and num steps {num_steps}")

        # --- Periodic Evaluation ---
        if episode % EVAL_FREQ == 0:
            avg_reward = evaluate_policy(policy, env)
            print(f"---------------------------------------")
            print(f"Evaluation after {episode} episodes: Average Reward: {avg_reward:.2f}")
            print(f"---------------------------------------")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(policy.state_dict(), 'best_reinforce_policy.pth')
                print(f"*** New best model saved with average reward: {best_avg_reward:.2f} ***")

    env.close()

def test_agent():
    env = LanelessEnv(render_mode='human', max_vehicles=1)
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

    