import gymnasium as gym
from laneless_env import LanelessEnv
import time

def run_laneless_env():
    # Instantiate the environment
    env = LanelessEnv(render_mode='human')

    # Reset the environment to get the initial state
    obs, info = env.reset()
    print("Environment reset.")

    terminated = False
    truncated = False
    episode_reward = 0

    # Run the simulation for a fixed number of steps
    for step in range(500):
        if terminated or truncated:
            print(f"Episode finished. Total reward: {episode_reward}")
            obs, info = env.reset()
            episode_reward = 0

        # Get a random action for each active agent
        actions = {agent_id: env.action_space[agent_id].sample() for agent_id in env.action_space.keys()}
        
        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Accumulate rewards
        episode_reward += sum(rewards.values())

        # Optional: print step info
        if step % 50 == 0:
            print(f"Step {step}, Active Agents: {len(obs)}, Total Reward: {sum(rewards.values()):.2f}")

        # The render call is handled within the environment's step method
        # when render_mode is 'human'

    # Close the environment
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    run_laneless_env()
