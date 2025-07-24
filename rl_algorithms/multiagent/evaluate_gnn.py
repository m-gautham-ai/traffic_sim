import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import time

from laneless_env import LanelessEnv
from gnn_policy import ActorCriticGNN

def evaluate_gnn():
    # --- Parameters ---
    model_path = 'ppo_gnn_best.pth'
    total_steps = 5000  # Run for a fixed number of steps
    num_vehicles = 20

    # --- Initialization ---
    # Initialize the environment in evaluation mode
    env = LanelessEnv(render_mode='human', max_vehicles=num_vehicles, evaluation_mode=True)
    policy = ActorCriticGNN(node_feature_dim=1, action_dim=2)
    
    # Load the trained model weights
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first to generate the model.")
        return
        
    policy.load_state_dict(torch.load(model_path))
    policy.eval() # Set the policy to evaluation mode

    # --- Run Evaluation ---
    obs, _ = env.reset()
    print("--- Starting Evaluation Run ---")
    
    for step in range(total_steps):
        graph_data, node_to_vehicle_map = env.get_graph()
        if not graph_data or graph_data.num_nodes == 0:
            # If no vehicles, step with an empty action dictionary to allow env to spawn more
            actions = {}
        else:
            # Get deterministic actions from the policy
            with torch.no_grad():
                dist, _ = policy(graph_data)
                # Map actions from node indices back to vehicle IDs
                actions = {node_to_vehicle_map[i]: dist.mean[i].cpu().numpy() for i in range(graph_data.num_nodes)}

        # Step the environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        env.render()
        time.sleep(1/60) # 60 FPS

        if (step + 1) % 100 == 0:
            print(f"Step: {step + 1}/{total_steps}")

    print("--- Evaluation Finished ---")
    env.close()

if __name__ == '__main__':
    evaluate_gnn()
