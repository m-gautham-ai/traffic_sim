import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from simulation import Simulation

class LanelessEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, max_vehicles=20):
        super().__init__()
        self.render_mode = render_mode

        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((1400, 800))
            pygame.display.set_caption("Laneless Environment")
            self.clock = pygame.time.Clock()

        self.simulation = Simulation(max_vehicles=max_vehicles, render_mode=render_mode)

        # Action space: [acceleration, steering] for each vehicle
        self.action_space = spaces.Dict()
        # Observation space: [speed, rel_x1, rel_y1, speed1, ...]
        self.observation_space = spaces.Dict()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_obs_dict = self.simulation.reset()
        self._update_spaces(initial_obs_dict)
        print(f"Environment reset. {len(self.simulation.sprites)} vehicles.")
        return self._get_obs(), self._get_info()

    def step(self, actions):
        obs, rewards, collided_vehicles, off_screen_vehicles, is_truncated_global = self.simulation.step(actions)
        self._update_spaces(obs)

        if self.render_mode == "human":
            self.render()

        # Determine which agents are terminated or truncated
        terminated_ids = {v.id for v in collided_vehicles}.union({v.id for v in off_screen_vehicles})
        
        terminated = {v.id: v.id in terminated_ids for v in self.simulation.sprites}
        truncated = {v.id: is_truncated_global for v in self.simulation.sprites}

        # An agent that is terminated should not also be truncated
        for agent_id in terminated_ids:
            if agent_id in truncated:
                truncated[agent_id] = False

        return obs, rewards, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == 'human':
            self.simulation.render(self.screen)
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

    def _update_spaces(self, obs_dict):
        """Dynamically update action and observation spaces based on active vehicles."""
        agent_ids = [v.id for v in self.simulation.sprites]
        self.action_space = spaces.Dict(
            {agent_id: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for agent_id in agent_ids}
        )
        self.observation_space = spaces.Dict(
            {agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs_dict[agent_id]),), dtype=np.float32) 
             for agent_id in agent_ids if agent_id in obs_dict}
        )

    def _get_obs(self):
        return self.simulation.get_observation()

    def _get_info(self):
        return {v.id: {} for v in self.simulation.sprites}

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()
