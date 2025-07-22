import random
import pygame
import math

class Simulation:
    def __init__(self, max_vehicles=20):
        self.speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}
        self.x = {'right': 0, 'down': (700, 750), 'left': 1400, 'up': (600, 650)}
        self.y = {'right': (350, 500), 'down': 0, 'left': (450, 500), 'up': 800}
        self.vehicle_types = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
        self.direction_numbers = {0: 'right'}
        self.default_stop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
        self.moving_gap = 25
        self.allowed_vehicle_types = {'car': True, 'bus': True, 'truck': True, 'bike': True}
        self.allowed_vehicle_types_list = [i for i, v in enumerate(self.allowed_vehicle_types.values()) if v]

        self.vehicles = {direction: [] for direction in self.direction_numbers.values()}
        self.sprites = pygame.sprite.Group()
        self.time_elapsed = 0
        self.frame_count = 0
        self.max_vehicles = max_vehicles
        self.next_vehicle_id = 0

        self.background = pygame.image.load('images/intersection.png')
        self.font = pygame.font.Font(None, 30)

    def reset(self):
        self.vehicles = {direction: [] for direction in self.direction_numbers.values()}
        self.sprites.empty()
        self.time_elapsed = 0
        self.frame_count = 0
        # Add some initial vehicles
        for _ in range(5):
            self.generate_vehicle()
        return self.get_observation()

    def generate_vehicle(self):
        if sum(len(v) for v in self.vehicles.values()) >= self.max_vehicles:
            return
        vehicle_type_idx = random.choice(self.allowed_vehicle_types_list)
        vehicle_type = self.vehicle_types[vehicle_type_idx]
        direction_number = random.choice(list(self.direction_numbers.keys()))
        direction = self.direction_numbers[direction_number]
        Vehicle(self, self.next_vehicle_id, vehicle_type, direction_number, direction)
        self.next_vehicle_id += 1

    def step(self, actions):
        self.time_elapsed += 1 / 30 # Assuming 30 FPS
        self.frame_count += 1

        # Spawn new vehicles periodically
        if self.frame_count % 30 == 0:
            self.generate_vehicle()

        rewards = {}
        # Apply actions to vehicles
        # Create a mapping from vehicle ID to vehicle object for quick lookup
        vehicle_map = {v.id: v for v in self.sprites}
        for vehicle_id, action in actions.items():
            if vehicle_id in vehicle_map:
                vehicle_map[vehicle_id].move(action)

        # --- Calculate Rewards ---
        for vehicle in self.sprites:
            rewards[vehicle.id] = self._calculate_reward(vehicle)

        # --- Handle Collisions and Off-Screen Vehicles ---
        self._handle_collisions_and_offscreen()
        
        observation = self.get_observation()
        terminated = self.is_terminated()
        truncated = self.is_truncated()

        return observation, rewards, terminated, truncated

    def get_observation(self):
        # Returns a dictionary of observations for each vehicle, keyed by vehicle ID
        observations = {}
        for vehicle in self.sprites:
            observations[vehicle.id] = vehicle.get_observation(self.sprites)
        return observations

    def _calculate_reward(self, vehicle):
        TARGET_SPEED = 2.5
        SAFE_DISTANCE = 60
        
        # 1. Speed reward
        speed_error = abs(vehicle.speed - TARGET_SPEED)
        speed_reward = math.exp(-0.5 * speed_error)

        # 2. Proximity penalty
        proximity_penalty = 0
        for other_vehicle in self.sprites:
            if vehicle is not other_vehicle:
                dist = math.sqrt((vehicle.x - other_vehicle.x)**2 + (vehicle.y - other_vehicle.y)**2)
                if dist < SAFE_DISTANCE:
                    proximity_penalty += -1.5 * (1 - (dist / SAFE_DISTANCE))
        
        return speed_reward + proximity_penalty

    def _handle_collisions_and_offscreen(self):
        off_screen_vehicles = {v for v in self.sprites if not self.background.get_rect().colliderect(v.rect)}
        
        collided_vehicles = set()
        for vehicle in self.sprites:
            if vehicle in off_screen_vehicles:
                continue
            # Check for collisions with other vehicles
            colliding_list = pygame.sprite.spritecollide(vehicle, self.sprites, False, pygame.sprite.collide_rect)
            if len(colliding_list) > 1:
                for collided in colliding_list:
                    if collided is not vehicle:
                        collided_vehicles.add(collided)
                collided_vehicles.add(vehicle)

        vehicles_to_remove = collided_vehicles.union(off_screen_vehicles)
        for vehicle in vehicles_to_remove:
            self.sprites.remove(vehicle)
            if vehicle in self.vehicles[vehicle.direction]:
                self.vehicles[vehicle.direction].remove(vehicle)

    def is_terminated(self):
        # e.g., if a major collision occurs
        return False

    def is_truncated(self):
        # e.g., if simulation time limit is reached
        return self.time_elapsed > 300

    def render(self, screen):
        screen.blit(self.background, (0, 0))
        for vehicle in self.sprites:
            screen.blit(vehicle.image, vehicle.rect)
        
        time_text = self.font.render(f"Time: {self.time_elapsed:.2f}s", True, (0,0,0))
        screen.blit(time_text, (1100, 50))
        vehicle_count_text = self.font.render(f"Vehicles: {len(self.sprites)}", True, (0,0,0))
        screen.blit(vehicle_count_text, (1100, 80))


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, simulation, vehicle_id, vehicle_class, direction_number, direction):
        super().__init__()
        self.sim = simulation
        self.id = vehicle_id
        self.vehicle_class = vehicle_class
        self.speed = self.sim.speeds[vehicle_class]
        self.direction_number = direction_number
        self.direction = direction
        self.crossed = 0

        # Randomize starting position
        if direction == 'right':
            self.x = self.sim.x[direction]
            self.y = random.randint(self.sim.y[direction][0], self.sim.y[direction][1])
        elif direction == 'left':
            self.x = self.sim.x[direction]
            self.y = random.randint(self.sim.y[direction][0], self.sim.y[direction][1])

        path = f"images/{direction}/{vehicle_class}.png"
        self.image = pygame.image.load(path)

        # Prevent spawn overlap
        if self.sim.vehicles[direction]:
            last_vehicle = self.sim.vehicles[direction][-1]
            if self.is_too_close(last_vehicle):
                if direction == 'right':
                    self.x = last_vehicle.x - self.image.get_rect().width - self.sim.moving_gap
                elif direction == 'left':
                    self.x = last_vehicle.x + self.image.get_rect().width + self.sim.moving_gap

        self.sim.vehicles[direction].append(self)
        self.sim.sprites.add(self)

        self.rect = self.image.get_rect(topleft=(self.x, self.y))

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
        can_move = True
        for other_vehicle in self.sim.vehicles[self.direction]:
            if self == other_vehicle: continue
            
            is_in_front = (self.direction == 'right' and self.x < other_vehicle.x) or \
                          (self.direction == 'left' and self.x > other_vehicle.x)
            
            if is_in_front and self.is_too_close(other_vehicle):
                dist_x = abs(self.x - other_vehicle.x)
                if dist_x < self.image.get_rect().width + self.sim.moving_gap:
                    can_move = False
                    break

        acceleration, steering = action
        self.speed += acceleration * 0.5
        self.speed = max(0, self.speed)

        if can_move:
            if self.direction == 'right': self.x += self.speed
            elif self.direction == 'left': self.x -= self.speed

            if self.direction in ['right', 'left']:
                self.y += steering * 2
                self.y = max(self.sim.y[self.direction][0], min(self.y, self.sim.y[self.direction][1]))
            
            self.rect.topleft = (self.x, self.y)