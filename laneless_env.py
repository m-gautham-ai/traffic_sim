import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from simulation import Simulation

class LanelessEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, max_vehicles=20, evaluation_mode=False):
        super().__init__()
        self.render_mode = render_mode

        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((1400, 800))
            pygame.display.set_caption("Laneless Environment")
            self.clock = pygame.time.Clock()

        self.simulation = Simulation(max_vehicles=max_vehicles, render_mode=render_mode, evaluation_mode=evaluation_mode)

        # Action space: [acceleration, steering] for each vehicle
        self.action_space = spaces.Dict()

    def get_graph(self, obs):
        # Get vehicle objects from the observation dictionary keys
        vehicles = [v for v in self.simulation.sprites if v.id in obs]
        graph_data = self.simulation.get_graph_observation(vehicles)
        if graph_data is None:
            return None, {}
        # The vehicle IDs are stored in the graph_data object. We create the map from it.
        node_to_vehicle_map = {i: vid for i, vid in enumerate(graph_data.vehicle_ids)}
        return graph_data, node_to_vehicle_map

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
        
        # In evaluation mode, the episode doesn't terminate until the step limit is reached
        terminated = {v.id: self.simulation.is_terminated() or v.id in terminated_ids for v in self.simulation.sprites}
        truncated = {v.id: is_truncated_global for v in self.simulation.sprites}

        # An agent that is terminated should not also be truncated
        for agent_id in terminated_ids:
            if agent_id in truncated:
                truncated[agent_id] = False

        return obs, rewards, collided_vehicles, off_screen_vehicles, is_truncated_global

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

    def get_safety_violations(self):
        return self.simulation.get_safety_violations()

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()
