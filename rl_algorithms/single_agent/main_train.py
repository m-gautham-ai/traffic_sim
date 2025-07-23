import sys
import os
import gymnasium as gym
import torch
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from laneless_env import LanelessEnv

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99):
        super(PolicyNetwork, self).__init__()

        self.linear1 = torch.nn.Linear(state_dim, 64)
        self.linear2 = torch.nn.Linear(64, 64)
        self.acceleration_layer = torch.nn.Linear(64, 1)
        self.steering_layer = torch.nn.Linear(64, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        acceleration = torch.tanh(self.acceleration_layer(x))
        steering = torch.tanh(self.steering_layer(x))
        return acceleration, steering   


    def update(self, states, actions, rewards):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        self.optimizer.zero_grad()
        if  not isinstance(states, torch.Tensor):
            states = torch.stack(states)
        if not isinstance(actions, torch.Tensor):
            actions = torch.stack(actions)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.stack(rewards)
        

        
env = LanelessEnv(render_mode='human', max_vehicles=1)
env.reset()
env.render()

policy = PolicyNetwork(state_dim=10, action_dim=2)


for episode in range(1):
    state, info = env.reset()
    print ( "the observation space is ", state )  
    states = []
    actions = []
    rewards = []

    episode_reward = 0
    for step in range(10):
        agent_ids = list(state.keys())
        actions = {}
        for agent_id in agent_ids:
            _observation = torch.tensor(state[agent_id])
            print( "the observation is ", _observation.shape )
            actions[agent_id] = policy(_observation)
            print ( "the action is ", actions[agent_id] )

        next_states, rewards, terminated,truncate, info = env.step(actions)
        print ( "the next state is ", len(next_states ))
        print ( "the reward is ", len(rewards) )


        



    
        
        # print ( "the observation is ", obs )

        # episode_reward += sum(reward.values())
        
        # env.render()
        # if terminated or truncate:
        #     break

    print(f"Episode {episode} finished with reward {episode_reward}")
