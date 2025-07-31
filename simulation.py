import random
import pygame
import math
import torch
import numpy as np
from torch_geometric.data import Data
from scipy.optimize import minimize

class Simulation:
    def __init__(self, max_vehicles=20, render_mode=None, evaluation_mode=False):
        self.render_mode = render_mode
        self.evaluation_mode = evaluation_mode

        self.speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}
        self.x = {'right': 0, 'down': (700, 750), 'left': 1400, 'up': (600, 650)}
        self.y = {'right': (350, 500), 'down': 0, 'left': (450, 500), 'up': 800}
        self.vehicle_types = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
        self.direction_numbers = {0: 'right'}
        self.default_stop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
        self.moving_gap = 25
        self.allowed_vehicle_types = {'car': True, 'bus': False, 'truck': False, 'bike': True}
        self.allowed_vehicle_types_list = [i for i, v in enumerate(self.allowed_vehicle_types.values()) if v]

        self.vehicles = {direction: [] for direction in self.direction_numbers.values()}
        self.sprites = pygame.sprite.Group()
        self.time_elapsed = 0
        self.frame_count = 0
        self.max_vehicles = max_vehicles
        self.road_width = 150 # y-range for 'right' direction (500-350)
        self.evaluation_mode = evaluation_mode
        self.next_vehicle_id = 0

        self.background = pygame.image.load('images/intersection.png')
        if render_mode == 'human':
            self.font = pygame.font.Font(None, 30)

    def reset(self):
        # Reset simulation state
        self.sprites.empty()
        self.vehicles = {direction: [] for direction in self.direction_numbers.values()}
        self.time_elapsed = 0
        self.frame_count = 0
        self.next_vehicle_id = 0

        # Deterministic initial placement for 'right' direction
        direction = 'right'
        direction_number = 0
        lane_min_y = 350 # Start of the road
        lane_max_y = 600 # End of the road
        available_height = lane_max_y - lane_min_y
        
        # Evenly space the vehicles within the lane
        spacing = available_height / self.max_vehicles

        for i in range(self.max_vehicles):
            vehicle_type_idx = random.choice(self.allowed_vehicle_types_list)
            vehicle_type = self.vehicle_types[vehicle_type_idx]
            
            # Calculate position
            pos_x = self.x[direction]
            pos_y = lane_min_y + (i * spacing)
            initial_pos = (pos_x, pos_y)

            Vehicle(self, self.next_vehicle_id, vehicle_type, direction_number, direction, initial_pos)
            self.next_vehicle_id += 1

        return self.get_observation()

    def generate_vehicle(self):
        min_spawn_distance = 40  # Pixels
        max_attempts = 50

        for _ in range(max_attempts):
            # Randomly select vehicle properties
            vehicle_type_idx = random.choice(self.allowed_vehicle_types_list)
            vehicle_type = self.vehicle_types[vehicle_type_idx]
            direction_number = 0  # Assuming 'right' direction
            direction = 'right'

            # Generate a candidate position
            lane_min_y = self.y[direction][0]
            lane_max_y = self.y[direction][1]
            candidate_x = self.x[direction]
            candidate_y = random.uniform(lane_min_y, lane_max_y)
            candidate_pos = np.array([candidate_x, candidate_y])

            # Check for safety
            is_safe = True
            for other_vehicle in self.sprites:
                other_pos = np.array([other_vehicle.x, other_vehicle.y])
                if np.linalg.norm(candidate_pos - other_pos) < min_spawn_distance:
                    is_safe = False
                    break
            
            if is_safe:
                Vehicle(self, self.next_vehicle_id, vehicle_type, direction_number, direction, candidate_pos)
                self.next_vehicle_id += 1
                return  # Successfully spawned

    def nonlinear_width_objective(self, w, w_desired, priorities, delays, alpha=1.0, beta=0.1, gamma=0.05, epsilon=1e-6):
        # Proportional Fairness: maximize sum(log(w)) -> minimize -sum(log(w))
        log_fairness_cost = -np.sum(np.log(w + epsilon))
        # Cost from deviation from desired width
        # EXP/PF Rule: Weight deviation cost by an exponential of the delay
        # This heavily prioritizes vehicles that have been waiting longer.
        delay_weight = np.exp(beta * delays)
        deviation_cost = np.sum(priorities * delay_weight * ((w - w_desired) / w_desired) ** 2)

        # Entropy-based penalty for equitable distribution based on priority
        # This encourages the width distribution to match the priority distribution.
        total_width = np.sum(w)
        p_w = w / (total_width + epsilon)

        total_priority = np.sum(priorities)
        p_prio = priorities / (total_priority + epsilon)

        # KL-divergence: D_KL(p_w || p_prio) = sum(p_w * log(p_w / p_prio))
        # We want to minimize this divergence.
        entropy_penalty = np.sum(p_w * (np.log(p_w + epsilon) - np.log(p_prio + epsilon)))

        return deviation_cost + alpha * log_fairness_cost + gamma * entropy_penalty

    def step(self, actions):
        random_number = random.randint(1, 10)

        self.time_elapsed += 1 / 30 # Assuming 30 FPS
        self.frame_count += 1

        # Spawn new vehicles periodically
        if self.frame_count % 10 == 0 and random_number > 3 :
            if len(self.sprites) < self.max_vehicles:
                self.generate_vehicle()

        rewards = {}
        # Apply actions to vehicles
        # Create a mapping from vehicle ID to vehicle object for quick lookup
        vehicle_map = {v.id: v for v in self.sprites}
        for vehicle_id, action in actions.items():
            if vehicle_id in vehicle_map:
                vehicle_map[vehicle_id].move(action)

        # --- Nonlinear Width Allocation ---
        active_vehicles = list(self.sprites)
        if active_vehicles:
            num_vehicles = len(active_vehicles)
            w_desired = np.full(num_vehicles, 2.0) # Desired width of 2.0 for all
            priorities = np.array([v.priority for v in active_vehicles])
            delays = np.array([self.frame_count - v.creation_time for v in active_vehicles])

            # Define bounds and constraints for the optimization
            bounds = [(0.5, 5.0) for _ in range(num_vehicles)]
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - self.road_width}

            # Run optimization
            result = minimize(
                self.nonlinear_width_objective, 
                np.array([v.width for v in active_vehicles]), 
                args=(w_desired, priorities, delays, 1.0, 0.1, 0.05),
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )

            if result.success:
                for i, vehicle in enumerate(active_vehicles):
                    vehicle.width = result.x[i]

        # --- Handle Collisions and Off-Screen Vehicles ---
        collided_vehicles, off_screen_vehicles = self._handle_collisions_and_offscreen()

        # In evaluation mode, replenish vehicles to maintain the max count
        # if self.evaluation_mode:
        #     while len(self.sprites) < self.max_vehicles:
        #         self.generate_vehicle()

        # Replenish vehicles to maintain the max count
        # while len(self.sprites) < self.max_vehicles:
        #     self.generate_vehicle()

        # --- Calculate Rewards ---
        # Start with the base reward for all active vehicles
        for vehicle in self.sprites:
            rewards[vehicle.id] = self._calculate_reward(vehicle)

        # Apply large penalty for collisions
        for vehicle in collided_vehicles:
            rewards[vehicle.id] = -100  # Large penalty for crashing

        # Apply large reward for finishing
        for vehicle in off_screen_vehicles:
            rewards[vehicle.id] = 200  # Increased reward for successfully exiting

        # --- Reward Shaping ---
        for vehicle in self.sprites:
            # 1. Survival bonus
            survival_bonus = 0.01
            # 2. Speed reward (encourage forward movement)
            speed_reward = vehicle.speed * 0.01
            # 3. Control penalty (penalize extreme actions)
            action_taken = actions.get(vehicle.id, 0.0)
            control_penalty = -0.01 * (action_taken**2)

            # Check if vehicle id is already in rewards (from crash/success)
            if vehicle.id in rewards:
                rewards[vehicle.id] += survival_bonus + speed_reward + control_penalty
            else:
                rewards[vehicle.id] = survival_bonus + speed_reward + control_penalty

        # 4. Near-miss penalty
        for vehicle1 in self.sprites:
            for vehicle2 in self.sprites:
                if vehicle1 is vehicle2:
                    continue
                distance = pygame.math.Vector2(vehicle1.rect.center).distance_to(pygame.math.Vector2(vehicle2.rect.center))
                if distance < vehicle1.rect.width * 1.5: # If vehicles are within 1.5 car lengths
                    if vehicle1.id in rewards: rewards[vehicle1.id] -= 0.1
                    if vehicle2.id in rewards: rewards[vehicle2.id] -= 0.1

        # --- Finalize Step ---
        observation = self.get_observation()
        is_truncated_global = self.is_truncated()

        # Return the sets of vehicles that have terminated, plus the global truncation flag
        return observation, rewards, collided_vehicles, off_screen_vehicles, is_truncated_global

    def get_observation(self):
        # Returns a dictionary of observations for each vehicle, keyed by vehicle ID
        observations = {}
        for vehicle in self.sprites:
            observations[vehicle.id] = vehicle.get_observation(self.sprites)
        return observations

    def get_graph_observation(self, vehicles=None):
        vehicles = list(self.sprites)
        if not vehicles:
            return None

        # 1. Node Features: [speed, width] for each vehicle
        features = [[v.speed.item() if hasattr(v.speed, 'item') else v.speed, v.width] for v in vehicles]
        node_features = torch.tensor(features, dtype=torch.float32)

        # Create a mapping from vehicle object to its index in the list
        vehicle_to_idx = {v: i for i, v in enumerate(vehicles)}

        # 2. Edge Index & Edge Features
        edge_list = []
        edge_features = []
        sensor_range = 200  # Same as before

        for i, vehicle1 in enumerate(vehicles):
            for j, vehicle2 in enumerate(vehicles):
                if i == j:
                    continue
                dist = math.sqrt((vehicle1.x - vehicle2.x)**2 + (vehicle1.y - vehicle2.y)**2)
                if dist < sensor_range:
                    edge_list.append([j, i]) # Edge from neighbor j to self i
                    relative_speed = vehicle1.speed - vehicle2.speed
                    scalar_relative_speed = relative_speed.item() if hasattr(relative_speed, 'item') else relative_speed
                    edge_features.append([dist, scalar_relative_speed])

        if not edge_list: # Handle case with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        # Create the PyG Data object
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        # We also need to know which vehicle corresponds to which node
        graph_data.vehicle_ids = [v.id for v in vehicles]

        return graph_data

    def _calculate_reward(self, vehicle):
        # TARGET_SPEED = 2.5
        # SAFE_DISTANCE = 60
        
        # # 1. Speed reward
        # speed_error = abs(vehicle.speed - TARGET_SPEED)
        # speed_reward = math.exp(-0.5 * speed_error)

        # # 2. Proximity penalty
        # proximity_penalty = 0
        # for other_vehicle in self.sprites:
        #     if vehicle is not other_vehicle:
        #         dist = math.sqrt((vehicle.x - other_vehicle.x)**2 + (vehicle.y - other_vehicle.y)**2)
        #         if dist < SAFE_DISTANCE:
        #             proximity_penalty += -1.5 * (1 - (dist / SAFE_DISTANCE))
        
        TARGET_SPEED = 3.5  # Further increased target speed
        SAFE_DISTANCE = 50
        STEP_PENALTY = -0.01

        # 1. Aggressive Speed Reward: Strongly incentivize hitting the target speed.
        speed_error = abs(vehicle.speed - TARGET_SPEED)
        speed_reward = max(0, 1 - speed_error / TARGET_SPEED) * 1.0 # Increased multiplier

        # 2. Aggressive Forward Movement Reward: Make progress the primary objective.
        forward_reward = vehicle.speed * 0.4 # Increased multiplier

        # 3. Proximity penalty: Penalizes getting too close to other vehicles.
        proximity_penalty = 0
        for other_vehicle in self.sprites:
            if vehicle is not other_vehicle:
                dist = math.sqrt((vehicle.x - other_vehicle.x)**2 + (vehicle.y - other_vehicle.y)**2)
                if dist < SAFE_DISTANCE:
                    # Penalty increases quadratically as the vehicle gets closer
                    proximity_penalty -= (1 - (dist / SAFE_DISTANCE))**2

        return speed_reward + forward_reward + proximity_penalty + STEP_PENALTY

    def _handle_collisions_and_offscreen(self):
        off_screen_vehicles = {v for v in self.sprites if not self.background.get_rect().colliderect(v.rect)}
        
        collided_vehicles = set()
        # Create a copy to iterate over, as we might modify the original group
        for vehicle in list(self.sprites):
            if vehicle in off_screen_vehicles or vehicle in collided_vehicles:
                continue

            # Check for collisions with other vehicles
            # False means do not kill the sprite upon collision
            colliding_list = pygame.sprite.spritecollide(vehicle, self.sprites, False, pygame.sprite.collide_rect)
            if len(colliding_list) > 1:
                for collided in colliding_list:
                    if collided is not vehicle:
                        collided_vehicles.add(collided)
                collided_vehicles.add(vehicle)

        # Now, remove the vehicles from the simulation groups
        vehicles_to_remove = collided_vehicles.union(off_screen_vehicles)
        for vehicle in vehicles_to_remove:
            self.sprites.remove(vehicle)
            if vehicle.direction in self.vehicles and vehicle in self.vehicles[vehicle.direction]:
                self.vehicles[vehicle.direction].remove(vehicle)

        return collided_vehicles, off_screen_vehicles

    def is_terminated(self):
        # In evaluation mode, the simulation never terminates on its own
        if self.evaluation_mode:
            return False
        
        # In normal (training) mode, you might have termination conditions.
        # For now, we assume termination is handled by the training script logic (e.g., on crash).
        return False

    def is_truncated(self):
        # e.g., if simulation time limit is reached
        # Truncate if the simulation runs for more than 300 steps (10 seconds)
        return self.frame_count > 300

    def render(self, screen):
        if self.render_mode == 'human':
            screen.blit(self.background, (0, 0))
            for vehicle in self.sprites:
                screen.blit(vehicle.image, vehicle.rect)
            
            time_text = self.font.render(f"Time: {self.time_elapsed:.2f}s", True, (0,0,0))
            screen.blit(time_text, (1100, 50))
            vehicle_count_text = self.font.render(f"Vehicles: {len(self.sprites)}", True, (0,0,0))
            screen.blit(vehicle_count_text, (1100, 80))

            frame_count_text = self.font.render(f"Frames: {self.frame_count}", True, (0,0,0))
            screen.blit(frame_count_text, (1100, 110))


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, simulation, vehicle_id, vehicle_class, direction_number, direction, initial_position):
        super().__init__()
        self.sim = simulation
        self.id = vehicle_id
        self.vehicle_class = vehicle_class
        self.speed = self.sim.speeds[vehicle_class]
        self.priority = {'bus': 3, 'truck': 2, 'car': 1, 'bike': 1}.get(vehicle_class, 1)
        self.direction_number = direction_number
        self.direction = direction
        self.creation_time = self.sim.frame_count
        self.crossed = 0
        self.x, self.y = initial_position[0], initial_position[1]
        self.width = 2.0 # Default width

        # 1. Load image and get dimensions
        path = f"images/{self.direction}/{self.vehicle_class}.png"
        self.image = pygame.image.load(path)
        img_rect = self.image.get_rect()

        # 2. Final clamp to ensure vehicle is fully on screen
        screen_width = self.sim.background.get_rect().width
        lane_min_y = self.sim.y['right'][0]
        lane_max_y = self.sim.y['right'][1] - img_rect.height
        self.x = max(0, min(self.x, screen_width - img_rect.width))
        self.y = max(lane_min_y, min(self.y, lane_max_y))

        # 3. Add to simulation and create the render rectangle
        self.sim.vehicles[self.direction].append(self)
        self.sim.sprites.add(self)
        self.rect = img_rect.move(int(self.x), int(self.y))

    def is_too_close(self, other_vehicle):
        if self.direction in ['right', 'left']:
            return abs(self.y - other_vehicle.y) < self.image.get_rect().height
        else:
            return abs(self.x - other_vehicle.x) < self.image.get_rect().width

    def get_observation(self, vehicles_group):
        sensor_range = 200
        num_neighbors = 3
        own_observation = [self.speed]

        neighbors = []
        for vehicle in vehicles_group:
            if vehicle is not self:
                dist = math.sqrt((self.x - vehicle.x)**2 + (self.y - vehicle.y)**2)
                if dist < sensor_range:
                    neighbors.append((dist, vehicle))
        
        neighbors.sort(key=lambda x: x[0])
        closest_neighbors = neighbors[:num_neighbors]

        neighbor_observation = []
        for i in range(num_neighbors):
            if i < len(closest_neighbors):
                _, neighbor = closest_neighbors[i]
                relative_x = (neighbor.x - self.x) / sensor_range
                relative_y = (neighbor.y - self.y) / sensor_range
                neighbor_observation.extend([relative_x, relative_y, neighbor.speed])
            else:
                neighbor_observation.extend([0, 0, 0])
        return own_observation + neighbor_observation

    def move(self, action):
        # The action is a single value representing acceleration
        acceleration = action.item() if hasattr(action, 'item') else action

        # Update speed based on acceleration action
        self.speed += acceleration * 0.5
        self.speed = max(0, self.speed) # Cannot have negative speed

        # Calculate potential new position
        dx = 0
        if self.direction == 'right':
            dx = self.speed
        elif self.direction == 'left':
            dx = -self.speed
        
        next_x = self.x + dx

        # Check for collision with vehicles in front
        can_move = True
        for other_vehicle in self.sim.vehicles[self.direction]:
            if self == other_vehicle: continue

            # Check if other_vehicle is in front of this one
            is_in_front = (self.direction == 'right' and self.x < other_vehicle.x) or \
                          (self.direction == 'left' and self.x > other_vehicle.x)

            if is_in_front:
                # Calculate the distance if this vehicle moves
                dist_if_move = abs(next_x - other_vehicle.x)
                safe_dist = self.image.get_rect().width + self.sim.moving_gap
                
                if dist_if_move < safe_dist:
                    can_move = False
                    self.speed = 0 # Stop if too close
                    break
        
        # Update position if movement is safe
        if can_move:
            self.x = next_x



        # CRITICAL: Update the rect position after all calculations
        self.rect.topleft = (self.x, self.y)