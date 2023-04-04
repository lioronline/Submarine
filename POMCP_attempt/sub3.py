import numpy as np
import math
import random
import pygame
pygame.init()
import copy
pygame.font.init()
pygame.mixer.init()

def get_map(filename):
    file = open(filename)
    mapgrid = np.array([list(map(int, s.strip().split(","))) for s in file.readlines()])
    return mapgrid

mapgrid = get_map("map.txt")
arr = np.array(mapgrid)

# Grid dimensions
M = mapgrid.shape[0]
N = mapgrid.shape[1]

# Locations
boat_index = np.where(arr == 1)
boat_x, boat_y = (boat_index[0][0], boat_index[1][0])
initial_x, initial_y = (boat_index[0][0], boat_index[1][0])
treasure_index = np.where(arr == 2)
treasures = {}
for i in range(len(treasure_index[0])):
    index = (treasure_index[0][i], treasure_index[1][i])
    treasures[i] = index

# Obstacle locations
obstacles_index = np.where((arr == 3) | (arr == 4) | (arr == 5))
obstacles = {}
for i in range(len(obstacles_index[0])):
    index = (obstacles_index[0][i], obstacles_index[1][i])
    obstacles[i] = index

obstacle_values = obstacles.values()

# Actions
actions = ['up', 'down', 'left', 'right', 'pickup', 'return']
action_to_move = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
    }
treasure_options = [0, 1, 2]
treasure_length = len(treasures)

treasures_found = []
treasures_picked = []

# State space
if treasure_length ==1:
        state_space = [(x, y, t1) for x in range(M) for y in range(N) for t1 in treasure_options]
elif treasure_length == 2:
        state_space = [(x, y, t1, t2) for x in range(M) for y in range(N) for t1 in treasure_options for t2 in treasure_options if not (t1 == 1 and t2 == 1)]
elif treasure_length == 3:
        state_space = [(x, y, t1, t2, t3) for x in range(M) for y in range(N) for t1 in treasure_options for t2 in treasure_options for t3 in treasure_options if not (t1 == 1 and t2 == 1) if not (t1 == 1 and t3 == 1) if not (t2 ==1 and t3 == 1)]

def transition_model(state, action, belief_state):
    x = state[0]
    y = state[1]
    new_x, new_y = x, y
    if action == 'up' and x > 0:
        new_x = x - 1
    elif action == 'down' and x < M - 1:
        new_x = x + 1
    elif action == 'left' and y > 0:
        new_y = y - 1
    elif action == 'right' and y < N - 1:
        new_y = y + 1
    new_cell_belief = belief_state[new_x][new_y]
    cell_belief = belief_state[x][y]
    if new_cell_belief['3'] == 1.0 or new_cell_belief['4'] == 1.0 or new_cell_belief['5'] == 1.0:
        return state
    t1 = state[2]

    #For t3, simply add another elif for treasures.get(2), and ensure for all pickups you ensure only 1 treasure is being picked up (all other treasures != 1). we changed that it need to be generaly in the location of a treasure and doesnt check a spesific one
    if treasure_length ==1:
        if action == 'pickup' and cell_belief['2'] == 1.0 and t1 == 0:
            t1 = 1
        elif action == 'return' and (x,y) == (boat_x, boat_y) and t1 == 1:
            t1 = 2
        else:
            x, y = new_x, new_y

    elif treasure_length == 2:
        t2 = state[3]
        if action == 'pickup' and cell_belief['2'] == 1.0 and t1 == 0 and t2 != 1:
            t1 = 1
        elif action == 'pickup' and cell_belief['2'] == 1.0 and t2 == 0 and t1 != 1:
            t2 = 1
        elif action == 'return' and (x, y) == (boat_x, boat_y):
            if t1 == 1:
                t1 = 2
            if t2 == 1:
                t2 = 2
        else:
            x, y = new_x, new_y

    elif treasure_length == 3:
        t2 = state [3]
        t3 = state[4]
        if action == 'pickup' and cell_belief['2'] == 1.0 and t1 == 0 and t2 != 1 and t3 != 1 :
            t1 = 1
        elif action == 'pickup' and cell_belief['2'] == 1.0 and t2 == 0 and t1 != 1 and t3 != 1:
            t2 = 1
        elif action == 'pickup' and cell_belief['2'] == 1.0 and t3 == 0 and t1 != 1 and t2 != 1:
            t3 = 1
        elif action == 'return' and (x, y) == (boat_x, boat_y):
            if t1 == 1:
                t1 = 2
            elif t2 == 1:
                t2 = 2
            elif t3 == 1:
                t3 = 2
        else:
            x, y = new_x, new_y

    if len(state) == 3:
        new_state = (x,y,t1)
    elif len(state) == 4:
        new_state = (x,y,t1,t2)
    elif len(state) == 5:
        new_state = (x,y,t1,t2,t3)
    return new_state

large_reward = 10
small_reward = 1

def reward_model(state, next_state, action, old_belief_state):
    if state == next_state:
        return -10
    x, y = state[0], state[1]
    if action in action_to_move:
        dx, dy = action_to_move[action]
        new_x, new_y = x + dx, y + dy
        holding = False
        if treasure_length == 1:
                if state[2] == 1:
                    holding = True
        elif treasure_length == 2:
                if state[2] == 1 or state[3] == 1:
                    holding = True
        elif treasure_length == 3:
                if state[2] == 1 and state[3] == 1 and state[4] == 1:
                    holding = True
        if not holding:
            if len(treasures_found) != 0:
                treasure_location = treasures_found[0]
                distance_reward = distance_to_location(x, y, treasure_location) - distance_to_location(new_x, new_y, treasure_location)
                return 100000 * distance_reward # if the boat found a treasure it should move towards it

            unobserved_neighbors = 0
            # Check for unobserved neighbors in all 8 directions
            for nx, ny in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
                neighbor_x, neighbor_y = new_x + nx, new_y + ny
                if in_map(neighbor_x, neighbor_y) and is_unobserved(old_belief_state[neighbor_x][neighbor_y]):
                    unobserved_neighbors += 1

            # Calculate the treasure probability in the direction of the move and the number of unobserved cells in the direction of the move
            treasure_sum = 0
            num_cells = 0
            unobserved_num = 0
            while in_map(new_x, new_y):
                cell = old_belief_state[new_x][new_y]
                treasure_sum += cell['2']
                num_cells += 1 # in case we want to do avrage
                if is_unobserved(cell):
                    unobserved_num += 1
                new_x, new_y = new_x + dx, new_y + dy
            action_reward = -0.1 + 1000 * treasure_sum  + 1000 * unobserved_neighbors + 1000 * unobserved_num # we might need to give more waight to some things

            
        else: # if the sub is holdimg a treasure it should move towards the boat
            distance_reward = distance_to_boat(x, y) - distance_to_boat(new_x, new_y)
            return 1000000 * distance_reward

    elif action == 'pickup':
        action_reward = 100

    elif action == 'return':
        action_reward = 1000  # Reward for returning the treasure to the boat
    return action_reward

def initialize_probability_belief_state(grid_rows, grid_cols):
    # Define the initial probabilities for each item
    initial_probabilities = {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0}

    # Create an empty grid with the given number of rows and columns
    grid = []
    for i in range(grid_rows):
        row = []
        for j in range(grid_cols):
            # Fill each cell in the grid with the initial probabilities
            row.append(initial_probabilities.copy())
        grid.append(row)

    row_len = len(grid)
    col_len = len(grid[0])
    for i in range(row_len):
        for j in range(col_len):
            # if i==0 refers to updating boat initial location
            if i == 0:
                if (i, j) == (boat_x, boat_y):
                    grid[i][j]['1'] = 1.0
                else:
                    grid[i][j]['4'] = 1.0
            elif (i,j) == (boat_x+1, boat_y-1) or (i,j) == (boat_x+1, boat_y) or (i,j) == (boat_x+1, boat_y+1):
                the_key=str(mapgrid[i][j])
                grid[i][j][the_key]= 1.0
            # if it is the bottom row, then there is a chance to find a treasure, rock or water
            elif i == row_len-1:
                
                grid[i][j]['2'] = 0.4
                grid[i][j]['3'] = 0.2
                grid[i][j]['0'] = 0.4
               
            # if it's not at the bottom, then there can be water, rock, or bombs surrounding it
            else:
                grid[i][j]['0'] = 0.8
                grid[i][j]['3'] = 0.15
                grid[i][j]['5'] = 0.05

    return grid

def observation_model(state):
    x = state[0]
    y = state[1]
    new_observation = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append('no')
        new_observation.append(row)
    #if we are at the end of the list I will leave the old value ('no')- new_state gives me a location [x][y]
    for i in range(3):
        if x+i-1>(M-1) or x-1+i<0:
            continue
        else:
            for j in range(3):
                if y+j-1>(N-1) or y+j-1<0:
                    continue
                else:
                    new_observation[i][j]=mapgrid[x-1+i][y-1+j]
                    
    return new_observation

def generate_noisy_observation(next_state, belief_state):
    x = next_state[0]
    y = next_state[1]
    new_observation = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append('no')
        new_observation.append(row)
    #if we are at the end of the list I will leave the old value ('no')- new_state gives me a location [x][y]
    for i in range(3):
        if x+i-1>(M-1) or x-1+i<0:
            continue
        else:
            for j in range(3):
                if y+j-1>(N-1) or y+j-1<0:
                    continue
                else:
                    cell_belief = belief_state[x-1+i][y-1+j]
                    new_observation[i][j]=sample_from_belief(cell_belief)
                    
    return new_observation

def update_belief_state(state, belief_state, new_observation):
    x = state[0]
    y = state[1]
    new_belief_state=belief_state
    for i in range(3):
        for j in range(3):
            if new_observation[i][j]=='no':
                continue
            else:
                the_key=str(new_observation[i][j])
                new_belief_state[x+i-1][y+j-1]['0']=0
                new_belief_state[x+i-1][y+j-1]['1']=0
                new_belief_state[x+i-1][y+j-1]['2']=0
                new_belief_state[x+i-1][y+j-1]['3']=0
                new_belief_state[x+i-1][y+j-1]['4']=0
                new_belief_state[x+i-1][y+j-1]['5']=0
                new_belief_state[x+i-1][y+j-1][the_key]= 1.0


    # for i in range(3):
    #     if x+i-2>9 or x-2+i<0:
    #         continue
    #     else:
    #         for j in range(3):
    #             if y+j-2>9 or y+j-2<0:
    #                 continue
    #         else:
    #             the_key=str(new_observation[i][j])
    #             if the_key=='0':
    #                 new_belief_state[x+i-2][y+j-2]['0']=0.85
    #                 new_belief_state[x+i-2][y+j-2]['1']=0
    #                 new_belief_state[x+i-2][y+j-2]['2']=0.01
    #                 new_belief_state[x+i-2][y+j-2]['3']=0.09
    #                 new_belief_state[x+i-2][y+j-2]['4']=0
    #                 new_belief_state[x+i-2][y+j-2]['5']=0.05
    #             if the_key=='1' or the_key=='4' or the_key=='2':
    #                 continue

    #             if the_key=='3':
    #                 new_belief_state[x+i-2][y+j-2]['0']=0.75
    #                 new_belief_state[x+i-2][y+j-2]['1']=0
    #                 new_belief_state[x+i-2][y+j-2]['2']=0.01
    #                 new_belief_state[x+i-2][y+j-2]['3']=0.19
    #                 new_belief_state[x+i-2][y+j-2]['4']=0
    #                 new_belief_state[x+i-2][y+j-2]['5']=0.05
    #             if the_key=='5':
    #                 new_belief_state[x+i-2][y+j-2]['0']=0.84
    #                 new_belief_state[x+i-2][y+j-2]['1']=0
    #                 new_belief_state[x+i-2][y+j-2]['2']=0.01
    #                 new_belief_state[x+i-2][y+j-2]['3']=0.15
    #                 new_belief_state[x+i-2][y+j-2]['4']=0
    #                 new_belief_state[x+i-2][y+j-2]['5']=0

    return new_belief_state

class Node:
    def __init__(self, state, belief_state, action, reward, parent=None):
        self.belief_state = belief_state
        self.state = state
        self.action = action
        self.reward = reward
        self.parent = parent
        self.visits = 0
        self.total_reward = 0
        self.children = []

    def add_child(self, child):
        self.children.append(child)
    
    def get_children(self):
        return self.children

    def has_children(self):
        return len(self.children) > 0
    
    def get_parent(self):
        return self.parent
    
    def get_belief_state(self): # not used for now
        return self.belief_state
    
    def get_reward(self):
        return self.reward
    
    def get_action(self):
        return self.action
    
    def get_state(self):
        return self.state
    
    def is_terminal(self):
        if treasure_length ==1:
            if self.state[:2] == (boat_x, boat_y) and self.state[2] == 2:
                return True
            else:
                return False
        elif treasure_length == 2:
            if self.state[:2] == (boat_x, boat_y) and self.state[2] == 2 and self.state[3] == 2:
                return True
            else:
                return False
        elif treasure_length == 3:
            if self.state[:2] == (boat_x, boat_y) and self.state[2] == 2 and self.state[3] == 2 and self.state[4] == 2:
                return True
            else:
                return False
            
    def get_legal_actions(self):
        return legal_actions(self.state, self.belief_state) # Need to define
    
    def update(self, new_reward):
        self.visits += 1
        self.total_reward += new_reward

    def get_num_visits(self):
        return self.visits
    
    def get_total_reward(self):
        return self.total_reward
    
def pomcp_select_action(state, belief_state, num_iterations, exploration_constant):
    belief_state_copy = copy.deepcopy(belief_state)
    # Define the root node of the search tree
    root_node = Node(state, belief_state_copy, None, 0)
    
    # Run POMCP for the specified number of iterations
    for _ in range(num_iterations):
        # Step 1: Selection
        # Start at the root node and recursively select the most promising child node using the UCT algorithm
        selected_node = root_node
        while selected_node.has_children():
            selected_node = uct_select_child(selected_node, exploration_constant)

        # Step 2: Expansion
        # Expand the selected node by generating a random action and a resulting child node
        if not selected_node.is_terminal():
            if len(selected_node.get_legal_actions()) == 0:
                print('reached a state with no legal actions')
                break
            if 'pickup' in selected_node.get_legal_actions():
                action = 'pickup'
            elif 'return' in selected_node.get_legal_actions():
                action = 'return'
            else:
                action = random.choice(selected_node.get_legal_actions())
            next_state = transition_model(state, action, belief_state_copy)
            next_belief_state = update_belief_state(next_state, belief_state_copy, generate_noisy_observation(next_state, belief_state_copy))
            next_reward = reward_model(state, next_state ,action, belief_state_copy)
            child_node = Node(next_state, next_belief_state, action, next_reward, selected_node)
            selected_node.add_child(child_node)

            if not child_node.is_terminal():
                # Step 3: Simulation
                # Simulate a rollout from the child node to a terminal state and return the total reward
                total_reward = simulate_rollout(child_node, next_state, next_belief_state, max_depth)
                # Step 4: Backpropagation
                # Backpropagate the total reward up the tree to update the statistics of all visited nodes
                while child_node is not None:
                    child_node.update(total_reward)
                    child_node = child_node.get_parent()

        # best_child_node = uct_select_child(root_node, exploration_constant)
    best_child_node = uct_select_child(root_node, exploration_constant)
    return best_child_node.get_action()

def uct_select_child(node, exploration_constant):
    # Compute the UCT value for each child node and select the one with the highest value
    best_child_node = None
    best_value = -float('inf')
    for child_node in node.get_children():
        if child_node.get_num_visits() == 0:
            value = float('inf')
        else:
          value = child_node.get_reward() / child_node.get_num_visits() + exploration_constant * math.sqrt(math.log(node.get_num_visits()) / child_node.get_num_visits())
        if value > best_value:
            best_child_node = child_node
            best_value = value
    return best_child_node

def simulate_rollout(node, state, belief_state, max_depth):
    belief_state_copy_2 = copy.deepcopy(belief_state)
    total_reward = 0
    depth = 0
    while not node.is_terminal() and depth < max_depth:
        action = rollout_policy(state, belief_state_copy_2)
        next_state = transition_model(state, action, belief_state_copy_2)
        next_belief_state = update_belief_state(next_state, belief_state_copy_2, generate_noisy_observation(next_state, belief_state_copy_2))
        next_reward = reward_model(state, next_state, action, belief_state_copy_2)
        total_reward += next_reward
        state = next_state
        belief_state_copy_2 = next_belief_state
        depth += 1
    return total_reward

def sample_from_belief(cell_belief):
    cell_values = []
    probabilities = []

    for value, probability in cell_belief.items():
        if probability > 0:
            cell_values.append(value)
            probabilities.append(probability)

    return np.random.choice(cell_values, p=probabilities) # here there might be a problem, if the functions returns the value of the index

def legal_actions(state, belief_state):
    x, y = state[0], state[1]
    legal_actions = []
    for action in actions:
        if action in ['up', 'down', 'left', 'right']:
            dx, dy = action_to_move[action]
            new_x, new_y = x + dx, y + dy
            if valid_coordinates(new_x, new_y, belief_state):
                legal_actions.append(action)
        elif action == 'pickup':
            cell_belief = belief_state[x][y]
            if cell_belief['2'] == 1.0:
                if any(treasure_status == 0 for treasure_status in state[2:]) and not any(treasure_status == 1 for treasure_status in state):
                    legal_actions.append(action)
        elif action == 'return':
            if (x, y) == (boat_x, boat_y) and any(treasure_status == 1 for treasure_status in state[2:]):
                legal_actions.append(action)
    return legal_actions

def distance_to_boat(x, y):
    return math.sqrt((x - boat_x)**2 + (y - boat_y)**2)

def distance_to_location(x, y, location):
    location_x = location[0]
    location_y = location[1]
    return math.sqrt((x - location_x)**2 + (y - location_y)**2)

def is_unobserved(cell_belief):
        return not any([value == 1.0 for value in cell_belief.values()])

def valid_coordinates(x, y, belief_state):
    if 0 <= x < M and 0 <= y < N:
        cell_belief = belief_state[x][y]
        if cell_belief['3'] == 1.0 or cell_belief['4'] == 1.0 or cell_belief['5'] == 1.0:
            return False
        else:
            return True
    return False

def in_map(x, y):
    if 0 <= x < M and 0 <= y < N:
        return True
    return False

# def rollout_policy(state, belief_state):
    best_action = None
    best_reward = float('-inf')
    
    for action in legal_actions(state, belief_state):
        next_state = transition_model(state, action, belief_state)
        reward = reward_model(state, next_state, action, belief_state)
        
        if reward > best_reward:
            best_action = action
            best_reward = reward
    
    return best_action

def rollout_policy(state, belief_state):
    best_action = None
    x , y = state[0], state[1]
    unexplored_locations = []
    for i in range(M):
        for j in range(N):
            if is_unobserved(belief_state[i][j]):
                unexplored_locations.append((i, j))
    if len(unexplored_locations) > 0:
        # Find the closest unexplored location using the Manhattan distance
        min_distance = float('inf')
        closest_location = None
        for location in unexplored_locations:
            distance = abs(x - location[0]) + abs(y - location[1])
            if distance < min_distance:
                min_distance = distance
                closest_location = location
    holding = False
    if treasure_length == 1:
            if state[2] == 1:
                holding = True
    elif treasure_length == 2:
            if state[2] == 1 or state[3] == 1:
                holding = True
    elif treasure_length == 3:
            if state[2] == 1 or state[3] == 1 or state[4] == 1:
                holding = True
    for action in legal_actions(state, belief_state):
        if action == 'pickup':
            return action
        elif action == 'return':
            return action
        else:
            next_state = transition_model(state, action, belief_state)
            next_x, next_y = next_state[0], next_state[1]
            if holding:
                min_distance = distance_to_boat(x, y)
                distance = distance_to_boat(next_x, next_x)
                if distance < min_distance:
                        min_distance = distance
                        best_action = action
            else:
                if len(treasures_found) != 0:
                    treasure_location = treasures_found[0]
                    min_distance = distance_to_location(x, y, treasure_location)
                    distance = distance_to_location(next_x, next_y, treasure_location)
                    if distance < min_distance:
                        min_distance = distance
                        best_action = action
                elif len(unexplored_locations) > 0:
                    min_distance = distance_to_location(x, y, closest_location)
                    distance = distance_to_location(next_x, next_y, closest_location)
                    if distance < min_distance:
                            min_distance = distance
                            best_action = action
    return best_action


# def rollout_policy(state, belief_state):
#     actions = legal_actions(state, belief_state)
#     action_rewards = []

#     x = state[0]
#     y = state[1]
#     for action in actions:
#         if action == 'pickup':
#             return action
#         elif action == 'return':
#             return action
#         else:
#             next_state = transition_model(state, action, belief_state)
#             action_reward = reward_model(state, next_state, action, belief_state)
#             action_rewards.append(action_reward)
#     total_reward = abs(sum(action_rewards))
#     if total_reward == 0:
#         if len(actions) == 0:
#             return 'no action found'
#         else:
#             return random.choices(actions)
#     normalized_rewards = [reward / total_reward for reward in action_rewards]
#     # Sort the actions by their rewards in descending order
#     sorted_actions = sorted(actions, key=lambda a: normalized_rewards[actions.index(a)], reverse=True)
#     # Find the highest reward
#     highest_reward = normalized_rewards[actions.index(sorted_actions[0])]
#     # Collect all actions with the highest reward
#     highest_reward_actions = [a for a in sorted_actions if normalized_rewards[actions.index(a)] == highest_reward]
#     # Randomly select an action from the highest reward actions
#     chosen_action = random.choice(highest_reward_actions)
#     return chosen_action

def is_goal_state(state):
    if treasure_length ==1:
        if state[:2] == (boat_x, boat_y) and state[2] == 2:
            return True
        else:
            return False
    elif treasure_length == 2:
        if state[:2] == (boat_x, boat_y) and state[2] == 2 and state[3] == 2:
            return True
        else:
            return False
    elif treasure_length == 3:
        if state[:2] == (boat_x, boat_y) and state[2] == 2 and state[3] == 2 and state[4] == 2:
            return True
        else:
            return False

def draw_map(screen, state, arr):
    # Define colors and images
    BLUE = (18,82,143)

    sub_image = pygame.image.load('submarine.png')
    sub_image = pygame.transform.scale(sub_image, (50, 50))
    treasure_image = pygame.image.load('treasure.png')
    treasure_image = pygame.transform.scale(treasure_image, (50, 50))
    sky_image = pygame.image.load('sky.png')
    sky_image = pygame.transform.scale(sky_image, (50, 50))
    boat_image = pygame.image.load('boat.png')
    boat_image = pygame.transform.scale(boat_image, (50, 50))
    boatandsub_image = pygame.image.load('boatandsub.png')
    boatandsub_image = pygame.transform.scale(boatandsub_image, (50, 50))
    bomb_image = pygame.image.load('bomb.png')
    bomb_image = pygame.transform.scale(bomb_image, (50, 50))
    seaweed_image = pygame.image.load('seaweed.png')
    seaweed_image = pygame.transform.scale(seaweed_image, (50, 50))

    # Clear the screen
    screen.fill(BLUE)
    
    # Drawing the map
    agent_x = state[0]
    agent_y = state[1]
    if state[:2] == (boat_x, boat_y):
        for i in range(M):
            for j in range(N):
                if arr[i][j] == 3:  # obstacle
                    screen.blit(seaweed_image, (j * 50, i * 50))
                elif arr[i][j] == 2:  # treasure
                    screen.blit(treasure_image, (j * 50, i * 50))
                elif arr[i][j] == 1:  # boat/agent
                    screen.blit(boat_image, (j * 50, i * 50))
                elif arr[i][j] == 4:  # sky
                    screen.blit(sky_image, (j * 50, i * 50))
                elif arr[i][j] == 5:  # marine mines
                    screen.blit(bomb_image, (j * 50, i * 50))
                screen.blit(boatandsub_image, (agent_y * 50, agent_x * 50))
    elif state[2] == 0:
        for i in range(M):
            for j in range(N):
                if arr[i][j] == 3:  # obstacle
                    screen.blit(seaweed_image, (j * 50, i * 50))
                elif arr[i][j] == 2:  # treasure
                    screen.blit(treasure_image, (j * 50, i * 50))
                elif arr[i][j] == 1:  # boat/agent
                    screen.blit(boat_image, (j * 50, i * 50))
                elif arr[i][j] == 4:  # sky
                    screen.blit(sky_image, (j * 50, i * 50))
                elif arr[i][j] == 5:  # marine mines
                    screen.blit(bomb_image, (j * 50, i * 50))
                screen.blit(sub_image, (agent_y * 50, agent_x * 50))
    else:
        for i in range(M):
            for j in range(N):
                if arr[i][j] == 3:  # obstacle
                    screen.blit(seaweed_image, (j * 50, i * 50))
                elif arr[i][j] == 2:  # treasure
                    screen.blit(treasure_image, (j * 50, i * 50))
                elif arr[i][j] == 1:  # boat/agent
                    screen.blit(boat_image, (j * 50, i * 50))
                elif arr[i][j] == 4:  # sky
                    screen.blit(sky_image, (j * 50, i * 50))
                elif arr[i][j] == 5:  # marine mines
                    screen.blit(bomb_image, (j * 50, i * 50))
                screen.blit(sub_image, (agent_y * 50, agent_x * 50))
    
    

    # Update the screen
    pygame.display.flip()

    # Delay for 500 milliseconds
    pygame.time.delay(0)

max_depth = 4
num_iterations = 1000
exploration_constant = 5
max_steps = 50 
step = 0
size = (N * 50, M * 50)
screen = pygame.display.set_mode(size)

if treasure_length ==1:
    current_state = (initial_x, initial_y, 0)
if treasure_length ==2:
    current_state = (initial_x, initial_y, 0, 0)
if treasure_length ==3:
    current_state = (initial_x, initial_y, 0, 0, 0)

current_belief_state = initialize_probability_belief_state(M, N)
actions_taken = []
while not is_goal_state(current_state) and step < max_steps:
    # Find the best action using POMCP
    best_action = pomcp_select_action(current_state, current_belief_state, num_iterations, exploration_constant)
    # if len(actions_taken) >= 2:
    #     if current_state[:2] == (boat_x, boat_y):
    #         best_action = best_action
    #     else:
    #         while best_action == actions_taken[-2]:
    #             best_action = pomcp_select_action(current_state, current_belief_state, num_iterations, exploration_constant)
    # actions_taken.append(best_action)
    print(best_action)
    # Execute the action and update the state
    next_state = transition_model(current_state, best_action, current_belief_state)
    print(next_state)
    # Update the belief state
    current_belief_state = update_belief_state(next_state, current_belief_state, observation_model(next_state))
    for i in range(N):
        cell_belief = current_belief_state[M-1][i]
        if cell_belief['2'] == 1.0:
            if not (M-1,i) in treasures_found:
                treasures_found.append((M-1,i))
    if best_action == 'pickup':
        x = next_state[0]
        y = next_state[1]
        arr[x][y] = 0
        treasures_found.remove((x,y))
    # Move to the next state
    current_state = next_state
    draw_map(screen, current_state, arr)
    step += 1

if is_goal_state(current_state):
    print("Goal reached!")
else:
    print("Max steps reached without finding the goal.")