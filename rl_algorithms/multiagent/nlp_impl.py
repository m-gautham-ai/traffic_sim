import gym
from gym import spaces
import numpy as np

class LaneFreeEnv(gym.Env):
    def __init__(self, num_agents=5, total_width=10.0):
        super(LaneFreeEnv, self).__init__()
        self.num_agents = num_agents
        self.total_width = total_width

        # Observation: lateral positions + widths for all agents
        self.observation_space = spaces.Box(low=-total_width/2, high=total_width/2, shape=(num_agents*2,), dtype=np.float32)

        # Action: lateral velocity for each agent
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents,), dtype=np.float32)

        # Initialize lateral positions centered evenly
        self.positions = np.linspace(-total_width/2 + total_width/(2*num_agents), total_width/2 - total_width/(2*num_agents), num_agents)
        self.widths = np.ones(num_agents) * (total_width / num_agents)

    def nonlinear_width_objective(self, w, w_desired, p, d, alpha=1.0, beta=1.0, epsilon=1e-6):
        # Proportional Fairness: maximize sum(log(w)) -> minimize -sum(log(w))
        log_fairness_cost = -np.sum(np.log(w + epsilon))
        
        # Cost from deviation from desired width
        deviation_cost = np.sum(((w - w_desired) / w_desired) ** 2)
        
        return deviation_cost + alpha * log_fairness_cost

    def step(self, actions):
        # Update positions based on lateral velocities (actions)
        self.positions += actions * 0.1  # time delta factor

        # Clip positions to road boundaries
        self.positions = np.clip(self.positions, -self.total_width/2, self.total_width/2)

        # Compute new widths using nonlinear optimization based on positions, priorities, etc.
        w_desired = np.ones(self.num_agents) * 2.0
        priorities = np.ones(self.num_agents)
        urgencies = np.abs(self.positions)  # example urgency based on distance from center

        bounds = [(0, self.total_width) for _ in range(self.num_agents)]
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_width})
        result = minimize(self.nonlinear_width_objective, w_desired, args=(w_desired, priorities, urgencies), method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            self.widths = result.x
        else:
            self.widths = w_desired

        obs = np.concatenate([self.positions, self.widths])
        reward = -np.sum(np.square(actions))  # example: penalize large lateral moves
        done = False
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.positions = np.linspace(-self.total_width/2 + self.total_width/(2*self.num_agents),
                                     self.total_width/2 - self.total_width/(2*self.num_agents),
                                     self.num_agents)
        self.widths = np.ones(self.num_agents) * (self.total_width / self.num_agents)
        return np.concatenate([self.positions, self.widths])

    def render(self):
        # Optionally: render positions and width envelopes using matplotlib
        pass

# Example usage:
# env = LaneFreeEnv()
# obs = env.reset()
# for _ in range(100):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
