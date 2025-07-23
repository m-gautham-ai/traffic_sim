import sys
import os
import gymnasium as gym

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from laneless_env import LanelessEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env



class SingleAgentWrapper(gym.Wrapper):
    """ A wrapper to convert the multi-agent style environment to a single-agent one. """
    def __init__(self, env):
        super().__init__(env)
        # Spaces will be defined after the first reset
        self.observation_space = None
        self.action_space = None
        self.agent_id = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.agent_id = list(obs.keys())[0]
        
        # Now that the env is reset, we can define the spaces
        self.observation_space = self.env.observation_space.spaces[self.agent_id]
        self.action_space = self.env.action_space.spaces[self.agent_id]
        
        return obs[self.agent_id], info

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step({self.agent_id: action})
        new_obs = obs.get(self.agent_id)
        
        # If the agent is done, provide a dummy observation to avoid errors
        if new_obs is None:
            new_obs = self.observation_space.sample()

        return new_obs, rewards[self.agent_id], terminated or truncated, info

def train():
    """Train an agent using Stable Baselines3."""
    # --- 1. Create and wrap the environment ---
    env = LanelessEnv(render_mode='human', max_vehicles=1)
    # It's good practice to check the custom environment
    # check_env(env) # This will fail due to the Dict space, so we wrap it first.
    
    wrapped_env = SingleAgentWrapper(env)

    # --- 2. Create and train the PPO model ---
    model = PPO("MlpPolicy", wrapped_env, verbose=1)
    model.learn(total_timesteps=25000)

    # --- 3. Save the model and close the environment ---
    model.save("ppo_laneless")
    wrapped_env.close()

    print("Training finished and model saved!")

if __name__ == '__main__':
    train()
