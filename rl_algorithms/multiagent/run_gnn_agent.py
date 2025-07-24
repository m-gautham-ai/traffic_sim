import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import time

from laneless_env import LanelessEnv
from gnn_policy import ActorCriticGNN

def run_gnn_agent():
    # --- Parameters ---
    model_path = 'ppo_gnn_best.pth'
    num_episodes = 10
    num_vehicles = 10

    # --- Initialization ---
    env = LanelessEnv(render_mode='human', max_vehicles=num_vehicles)
    policy = ActorCriticGNN(node_feature_dim=1, action_dim=2)
    
    # Load the trained model weights
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first to generate the model.")
        return
        
    policy.load_state_dict(torch.load(model_path))
    policy.eval() # Set the policy to evaluation mode

    # --- Run Episodes ---
    for i in range(num_episodes):
        obs, _ = env.reset()
        print(f"--- Episode {i+1} ---")
        done = False
        while not done:
            graph_data = env.get_graph()
            if not graph_data or graph_data.num_nodes == 0:
                break

            # Get deterministic actions from the policy
            with torch.no_grad():
                dist, _ = policy(graph_data)
                actions_tensor = dist.mean
            actions_dict = {vid: act.cpu().detach().numpy() for vid, act in zip(graph_data.vehicle_ids, actions_tensor)}

            # Step the environment
            next_obs, _, _, truncated, _ = env.step(actions_dict)
            
            # Check for termination
            done = not next_obs or any(truncated.values())
            obs = next_obs
            
            # Render and sleep
            env.render()
            time.sleep(1/30) # 30 FPS

    env.close()

if __name__ == '__main__':
    run_gnn_agent()
