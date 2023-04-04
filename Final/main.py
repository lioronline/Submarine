import numpy as np
import matplotlib.pyplot as plt
import pygame
import math
import random
pygame.init()
pygame.font.init()
pygame.mixer.init()

# Building the map
def get_map(filename):
    file = open(filename)
    mapgrid = np.array([list(map(int, s.strip().split(","))) for s in file.readlines()])
    return mapgrid

# Create a list with elements "map0.txt" to "map9.txt"
file_list = [f"map{i}.txt" for i in range(10)]

# Choose a random string from the list
random_file = random.choice(file_list)
print("Chosen map:", end = " ")
print(random_file)

# Translating map to array
mapgrid = get_map(random_file)
arr = np.array(mapgrid)

# Defining sounds
pickup_sound = pygame.mixer.Sound("coin.wav")
waves_sound = pygame.mixer.Sound("waves.wav")
game_sound = pygame.mixer.Sound("game.wav")
success_sound = pygame.mixer.Sound("success.wav")

# Grid dimensions
M = mapgrid.shape[0]
N = mapgrid.shape[1]

# Defining end message
end_image = pygame.image.load('end.png')
end_image = pygame.transform.scale(end_image, (N * 50, M * 50))

# Locations
boat_index = np.where(arr == 6)
sub_x, sub_y = (boat_index[0][0], boat_index[1][0])
initial_x, initial_y = (boat_index[0][0], boat_index[1][0]) # This will be used to check if the sub is at the boat
sub_x, sub_y = initial_x, initial_y # initialize sub location

# Checking the amount of treasure in the map, the agent knows this from the start
treasure_index = np.where(arr == 2)
treasure_length = len(treasure_index[0])

# Actions
actions = ['up', 'down', 'left', 'right', 'pickup', 'return']
action_to_move = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
    }

# Define the range of rows and columns that the sub can observe
row_range = range(max(sub_x-1, 0), min(sub_x+1, len(arr)-1)+1)
col_range = range(max(sub_y-1, 0), min(sub_y+1, len(arr[0])-1)+1)

#observable map
obs_map = np.copy(arr)
arr[sub_x][sub_y] = 1 #so arr represents the map without the boat
for i in range(M):
    for j in range(N):
        if i not in row_range or j not in col_range:
            obs_map[i][j] = 9 #9 represents unknown spot
            
# Observed 0bstacle locations
obstacles_index = np.where((obs_map == 3)| (obs_map == 4) | (obs_map == 5))
obstacles = {}
for i in range(len(obstacles_index[0])):
    index = (obstacles_index[0][i], obstacles_index[1][i])
    obstacles[i] = index
obstacle_values = obstacles.values()

#uptates the observable map
def update_obs_map(action, arr, obs_map, sub_x, sub_y,M,N):
        row_range = range(max(sub_x-1, 0), min(sub_x+1, len(arr)-1)+1)
        col_range = range(max(sub_y-1, 0), min(sub_y+1, len(arr[0])-1)+1)
        if action == 'right': 
            for i in row_range:
                if sub_y+3<=N:
                    obs_map[i][sub_y+2] = arr[i][sub_y+2]
            if sub_y<N:   
                if obs_map[sub_x][sub_y+1]==2:
                    obs_map[sub_x][sub_y] = 0
                    obs_map[sub_x][sub_y+1] = 6
                else:
                    obs_map[sub_x][sub_y] = obs_map[sub_x][sub_y+1]
                    obs_map[sub_x][sub_y+1] = 6
                    sub_y+=1
        elif action == 'left': 
            for i in row_range:
                if sub_y-2>=0:
                    obs_map[i][sub_y-2] = arr[i][sub_y-2]
            if sub_y>0:  
                if obs_map[sub_x][sub_y-1]==2:
                    obs_map[sub_x][sub_y] = 0
                    obs_map[sub_x][sub_y-1] = 6
                else:  
                    obs_map[sub_x][sub_y] = obs_map[sub_x][sub_y-1]
                    obs_map[sub_x][sub_y-1] = 6
                    sub_y-=1
        elif action == 'up': 
            for j in col_range:
                if sub_x-2>=0:
                    obs_map[sub_x-2][j] = arr[sub_x-2][j]
            if sub_x>0:    
                if obs_map[sub_x-1][sub_y]==1:
                    obs_map[sub_x][sub_y] = 0
                    obs_map[sub_x-1][sub_y] = 6
                else:
                    obs_map[sub_x][sub_y] = obs_map[sub_x-1][sub_y]
                    obs_map[sub_x-1][sub_y] = 6
                    sub_x-=1
        elif action == 'down':
            for j in col_range:
                if sub_x+2<M:
                    obs_map[sub_x+2][j] = arr[sub_x+2][j]
            if sub_x<M-1:    
                if obs_map[sub_x+1][sub_y]==2:
                    obs_map[sub_x][sub_y] = 0
                    obs_map[sub_x+1][sub_y] = 6
                elif (sub_x, sub_y) == (initial_x, initial_y):
                    obs_map[sub_x][sub_y] = 1
                    obs_map[sub_x+1][sub_y] = 6
                else:
                    obs_map[sub_x][sub_y] = obs_map[sub_x+1][sub_y]
                    obs_map[sub_x+1][sub_y] = 6
                    sub_x+=1

def distance_to_location(x, y, location):
    location_x = location[0]
    location_y = location[1]
    return math.sqrt((x - location_x)**2 + (y - location_y)**2)

def in_map(x, y):
    if 0 <= x < M and 0 <= y < N:
        return True
    return False

# Checks if a cell is serounded by obstacles so that the find_closest_unknown knows not to search there
def check_neighbors(obs_map, i, j):
    neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
    for n in neighbors:
        if not (0 <= n[0] < M and 0 <= n[1] < N):
            # neighbor is out of bounds
            continue
        if obs_map[n[0], n[1]] not in (3, 4, 5, 9):
            # neighbor has a value other than 3, 4, 5, or 9
            return False
    return True

def find_closest_unknown(obs_map, sub_x, sub_y):
    #searches for the action that will make the distatnce between the sub and the closest unknown cell the smallest
    closest_nine_dist = float('inf')
    closest_nine_dir = None
    for i in range(M):
        for j in range(N):
            if obs_map[i,j] == 9:
                dist = abs(i - sub_x) + abs(j - sub_y)
                if dist < closest_nine_dist:
                    closest_nine_dist = dist
                    if not check_neighbors(obs_map, i, j): # if the cell has neighbors that are not obstacles or unknows so that it doesnt get stuck
                        for action in action_to_move:
                            new_x = sub_x + action_to_move[action][0]
                            new_y = sub_y + action_to_move[action][1]
                            if not (new_x,new_y) in obstacle_values and in_map(new_x,new_y):
                                if distance_to_location(new_x,new_y,(i,j)) < distance_to_location(sub_x,sub_y,(i,j)):
                                    closest_nine_dir = action
    if closest_nine_dir == None:
        return random.choice(legal_actions(state, obstacle_values)) # if there is no action that will make it closer to an unknown cell then it gets a random action (doesnt work so well)
    else:
        return closest_nine_dir

def opposite_action(action):
    if action == 'up':
        opposite_action = 'down'
    elif action == 'down':
        opposite_action = 'up'
    elif action == 'left':
        opposite_action = 'right'
    elif action == 'right':
        opposite_action = 'left'
    return opposite_action

def return_to_boat(actions_taken): # goes through the actions that have been taken to that point and returns the sub to below the boat
    counter = -1
    sub_index = np.where(obs_map == 6)
    sub_x, sub_y = (sub_index[0][0], sub_index[1][0])
    while not (sub_x, sub_y) == (initial_x + 1, initial_y):
        action = actions_taken[counter]
        action = opposite_action(action)
        update_obs_map(action, arr, obs_map, sub_x, sub_y,M,N)
        sub_index = np.where(obs_map == 6)
        sub_x, sub_y = (sub_index[0][0], sub_index[1][0])
        draw_map(screen, obs_map)
        counter -= 1

def next_step(row_range,col_range,obs_map,M,N,previous_step):
    alpha=0.4
    beta=0.3
    gamma=0.1
    policy = ['up','down','left','right']
    values = [0,0,0,0]
    obstacles_index = np.where((obs_map == 3) |(obs_map == 4)| (obs_map == 5))
    obstacles = {}
    for i in range(len(obstacles_index[0])):
        index = (obstacles_index[0][i], obstacles_index[1][i])
        obstacles[i] = index
    obstacle_values = obstacles.values()
    action_options = legal_actions(state, obstacle_values)
    counter=[0,0,0,0]
    for i in range(4):
        border = False
        if policy[i]=='up' and sub_x>1:
            for j in col_range:
                    if obs_map[sub_x-2][j]==9:
                        counter[i]+=1
            if (sub_x-1, sub_y) in obstacle_values:
                values[i]=-1
                continue
            else:
                if counter[i]==0 and previous_step=='down':
                    values[i]=-0.5
                    continue
        if policy[i]=='down' and sub_x<M-2:
            for j in col_range:
                    if obs_map[sub_x+2][j]==9:
                        counter[i]+=1
            if (sub_x+1, sub_y) in obstacle_values:
                values[i]=-1
                continue
            else:
                if counter[i]==0 and previous_step=='up':
                    values[i]=-0.5
                    continue
        if policy[i]=='right' and sub_y<N-2:
            for j in row_range:
                    if obs_map[j][sub_y+2]==9:
                        counter[i]+=1
            if (sub_x, sub_y+1) in obstacle_values:
                values[i]=-1
                continue
            else:
                if counter[i]==0 and previous_step=='left':
                    values[i]=-0.5
                    continue
        if policy[i]=='left' and sub_y>1:
            for j in row_range:
                    if obs_map[j][sub_y-2]==9:
                        counter[i]+=1
            if (sub_x, sub_y-1) in obstacle_values:
                values[i]=-1
                continue
            else:
                if counter[i]==0 and previous_step=='right':
                    values[i]=-0.5
                    continue
        values[i] = alpha*(1 if policy[i]=='down' else 0)+beta*counter[i]
    best_action_index = values.index(max(values))
    zeros=0
    for num in counter:
        if num == 0:
            zeros += 1
    if zeros==4:
        return find_closest_unknown(obs_map, sub_x, sub_y)
    else:
        best_action = policy[best_action_index]
        if best_action in action_options: #checks if the action chosen is a legal one and if not chooses a random one from the legal options
            return best_action
        else:
            return random.choice(action_options)   
 
# Visualizing using Pygame
def draw_map(screen, obs_map):
    # Define colors and images
    BLACK = (0,0,0)
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
    screen.fill(BLACK)
    sub_index_copy = np.where(obs_map == 6)
    sub_x_copy, sub_y_copy = (sub_index_copy[0][0], sub_index_copy[1][0])
    # Drawing the map
    if sub_x_copy==initial_x and sub_y_copy==initial_y:
        for i in range(M):
            for j in range(N):
                if obs_map[i][j] == 3:  # obstacle
                    screen.blit(seaweed_image, (j * 50, i * 50))
                elif obs_map[i][j] == 2:  # treasure
                    screen.blit(treasure_image, (j * 50, i * 50))
                elif obs_map[i][j] == 6:  # boat/agent
                    screen.blit(boatandsub_image, (j * 50, i * 50))
                elif obs_map[i][j] == 4:  # sky
                    screen.blit(sky_image, (j * 50, i * 50))
                elif obs_map[i][j] == 5:  # marine mines
                    screen.blit(bomb_image, (j * 50, i * 50))
                elif obs_map[i][j] == 0:  # water
                    pygame.draw.rect(screen, BLUE, (j * 50, i * 50, 50, 50))
        screen.blit(boatandsub_image, (initial_y * 50, initial_x * 50))

    else:
        for i in range(M):
            for j in range(N):
                if obs_map[i][j] == 3:  # obstacle
                    screen.blit(seaweed_image, (j * 50, i * 50))
                elif obs_map[i][j] == 2:  # treasure
                    screen.blit(treasure_image, (j * 50, i * 50))
                elif obs_map[i][j] == 6:  # boat/agent
                    screen.blit(sub_image, (j * 50, i * 50))
                elif obs_map[i][j] == 4:  # sky
                    screen.blit(sky_image, (j * 50, i * 50))
                elif obs_map[i][j] == 5:  # marine mines
                    screen.blit(bomb_image, (j * 50, i * 50))
                elif obs_map[i][j] == 0:  # water
                    pygame.draw.rect(screen, BLUE, (j * 50, i * 50, 50, 50))
        screen.blit(boat_image, (initial_y * 50, initial_x * 50))

    # Update the screen
    pygame.display.flip()

    # Delay for 500 milliseconds
    pygame.time.delay(500)
 
 # Reward function

# Reward function
large_reward = 10
small_reward = 1

def reward(state, action):
    if len(state) == 3:
        x,y,t1 = state
        next_x, next_y, next_t1 = next_state(state, action)
        if action == 'pickup' and (x, y) in obs_treasure.values() and t1 == 0:
            return small_reward
        elif action == 'return' and (x, y) == (initial_x, initial_y) and t1 == 1:
            return large_reward
        elif (x,y,t1) == (next_x,next_y,next_t1) and not t1 == 2:
            return -1 * large_reward
        return -0.1
    elif len(state) == 4:
        x, y, t1, t2 = state
        next_x, next_y, next_t1, next_t2 = next_state(state, action)
        if action == 'pickup' and (x, y) in obs_treasure.values() and t1 == 0 and t2 != 1:
            return small_reward
        elif action == 'pickup' and (x, y) in obs_treasure.values() and t2 == 0 and t1 != 1:
            return small_reward
        elif action == 'return' and (x, y) == (initial_x, initial_y) and (t1 == 1 or t2 == 1):
            return large_reward
        elif (x,y,t1,t2) == (next_x,next_y,next_t1,next_t2) and not (t1 == 2 and t2 == 2):
            return -1 * large_reward
        return -0.1
    elif len(state) == 5:
        x, y, t1, t2, t3 = state
        next_x, next_y, next_t1, next_t2, next_t3 = next_state(state, action)
        if action == 'pickup' and (x, y) in obs_treasure.values() and t1 == 0 and t2 != 1 and t3 != 1:
            return small_reward
        elif action == 'pickup' and (x, y) in obs_treasure.values() and t2 == 0 and t1 != 1 and t3 != 1:
            return small_reward
        elif action == 'pickup' and (x, y) in obs_treasure.values() and t3 == 0 and t1 != 1 and t2 != 1:
            return small_reward
        elif action == 'return' and (x, y) == (initial_x, initial_y) and (t1 == 1 or t2 == 1 or t3 == 1):
            return large_reward
        elif (x,y,t1,t2,t3) == (next_x,next_y,next_t1,next_t2,next_t3) and not (t1 == 2 and t2 == 2 and t3 == 2):
            return -1 * large_reward
        return -0.1
 
 # Modified Transition function

def next_state(state, action):
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
    
    if (new_x, new_y) in obstacle_values:
        return state
    
    t1 = state[2]

    if treasure_length ==1:
        if action == 'pickup' and (x,y) in obs_treasure.values() and t1 == 0:
            t1 = 1
        elif action == 'return' and (x,y) == (initial_x, initial_y) and t1 == 1:
            t1 = 2
        else:
            x, y = new_x, new_y

    elif treasure_length == 2:
        t2 = state[3]
        if action == 'pickup' and (x, y) in obs_treasure.values() and t1 == 0 and t2 != 1:
            t1 = 1
        elif action == 'pickup' and (x, y) in obs_treasure.values() and t2 == 0 and t1 != 1:
            t2 = 1
        elif action == 'return' and (x, y) == (initial_x, initial_y):
            if t1 == 1:
                t1 = 2
            if t2 == 1:
                t2 = 2
        else:
            x, y = new_x, new_y

    elif treasure_length == 3:
        t2 = state [3]
        t3 = state[4]
        if action == 'pickup' and (x, y) in obs_treasure.values() and t1 == 0 and t2 != 1 and t3 != 1 :
            t1 = 1
        elif action == 'pickup' and (x, y) in obs_treasure.values() and t2 == 0 and t1 != 1 and t3 != 1:
            t2 = 1
        elif action == 'pickup' and (x, y) in obs_treasure.values() and t3 == 0 and t1 != 1 and t2 != 1:
            t3 = 1
        elif action == 'return' and (x, y) == (initial_x, initial_y):
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
     
def valid_coordinates(x, y, obstacle_values):
    if 0 <= x < M and 0 <= y < N:
        if not (x,y) in obstacle_values:
            return True
        else:
            return False
    return False

def legal_actions(state,obstacle_values): # gives out the legal actions the agent can make based on the state and belief state
    x, y = state[0], state[1]
    legal_actions = []
    for action in actions:
        if action in ['up', 'down', 'left', 'right']:
            dx, dy = action_to_move[action]
            new_x, new_y = x + dx, y + dy
            if valid_coordinates(new_x, new_y, obstacle_values):
                legal_actions.append(action)
        elif action == 'pickup':
            if (x,y) in obs_treasure.values():
                if any(treasure_status == 0 for treasure_status in state[2:]) and not any(treasure_status == 1 for treasure_status in state):
                    legal_actions.append(action)
        elif action == 'return':
            if (x, y) == (initial_x, initial_y) and any(treasure_status == 1 for treasure_status in state[2:]):
                legal_actions.append(action)
    return legal_actions

# Discount factor
gamma = 0.99   
        
def policy_iteration(obstacle_values):
    print("Formulating plan")
    policy = np.zeros(len(state_space), dtype=object)
    value = np.zeros(len(state_space))
    for i, state in enumerate(state_space):
        policy[i] = np.random.choice(actions)
    while True:
        # Policy evaluation
        while True:
            delta = 0
            for i, state in enumerate(state_space):
                old_value = value[i]
                action = policy[i]
                value[i] = reward(state, action) + gamma * value[state_space.index(next_state(state, action))]
                delta = max(delta, abs(old_value - value[i]))
            if delta < 1e-6:
                break

        # Policy improvement
        policy_stable = True
        for i, state in enumerate(state_space):
            old_action = policy[i]
            best_action_value = -np.inf
            best_action = None
            for action in actions:
                temp_value = reward(state, action) + gamma * value[state_space.index(next_state(state, action))]
                if temp_value > best_action_value:
                    best_action_value = temp_value
                    best_action = action
            policy[i] = best_action
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break

    return policy, value

def execute_policy(optimal_policy, initial_state, num_treasures_left):
    state = initial_state
    plan = []  # sequence of (state, action) pairs
    total_reward = 0.0
    while True:
        sub_index = np.where(obs_map == 6)
        sub_x, sub_y = (sub_index[0][0], sub_index[1][0])
        action = optimal_policy[state_space.index(state)]
        plan.append((state, action))
        total_reward = total_reward + reward(state, action)
        state= next_state(state, action)
        x, y = state[0], state[1]
        if obs_treasure.get(0) == (x, y):  # Plays coin sound
            pickup_sound.play()
            pickup_sound.set_volume(1)
        if action == 'pickup':  # Deletes treasure from map
            arr[x][y] = 0 # delets the treasure from the "real map"
            num_treasures_left -= 1
        update_obs_map(action, arr, obs_map, sub_x, sub_y,M,N)
        draw_map(screen, obs_map)
        if len(treasure_index[0]) == 1:
            if state[:2] == (initial_x, initial_y) and state[2] == 2:  # reached the boat with the treasure
                plan.append((state, 'done'))
                break
        elif len(treasure_index[0]) == 2:
            if state[:2] == (initial_x, initial_y) and state[2] == 2 and state[3] == 2:  # reached the boat with the treasure
                plan.append((state, 'done'))
                break
        elif len(treasure_index[0]) == 3:
            if state[:2] == (initial_x, initial_y) and state[2] == 2 and state[3] == 2 and state[4] == 2:  # reached the boat with the treasure
                plan.append((state, 'done'))
                break
        if action == 'return':
            print('returned treasure')
            final_state = (state)
            return plan, total_reward, final_state, num_treasures_left

    final_state = (state)
    return plan, total_reward, final_state, num_treasures_left
    
previous_step=''
obs_treasure_index = np.where(obs_map == 2)  
obs_treasure = {}
num_treasures_left = len(obs_treasure_index[0])

 # Initialize the Pygame screen
size = (N * 50, M * 50)
screen = pygame.display.set_mode(size)
draw_map(screen, obs_map)

# Playing the sounds
waves_sound.play(-1)
game_sound.play(-1)
waves_sound.set_volume(0.1)
game_sound.set_volume(0.2)


treasure_options = [0, 1, 2] # these are the value that the treasure information in the state can take
num_runs = len(treasure_index[0]) # the amout of treasure to find
for i in range(num_runs):
    treasure_index = np.where(arr == 2) #checks the amout of treasure left in the map so that after a treasure has been returned we can work with a smaller state space for the MDP
    treasure_length = len(treasure_index[0]) 
    actions_taken = []

    if treasure_length ==1:
        state_space = [(x, y, t1) for x in range(M) for y in range(N) for t1 in treasure_options]
    elif treasure_length == 2:
            state_space = [(x, y, t1, t2) for x in range(M) for y in range(N) for t1 in treasure_options for t2 in treasure_options if not (t1 == 1 and t2 == 1)]
    elif treasure_length == 3:
            state_space = [(x, y, t1, t2, t3) for x in range(M) for y in range(N) for t1 in treasure_options for t2 in treasure_options for t3 in treasure_options if not (t1 == 1 and t2 == 1) if not (t1 == 1 and t3 == 1) if not (t2 ==1 and t3 == 1)]
    
    if len(treasure_index[0]) == 1:
        state = (sub_x, sub_y, 0)  # start state
    elif len(treasure_index[0]) == 2:
        state = (sub_x, sub_y, 0, 0)  # start state
    elif len(treasure_index[0]) == 3:
        state = (sub_x, sub_y, 0, 0, 0)  # start state
    #treasure search
    while num_treasures_left==0:
        action = next_step(row_range,col_range,obs_map,M,N,previous_step)
        if action == None: # not really used now because the functions return a random action if None but cvan be an option to get out of being stuck
            return_to_boat(actions_taken)
            actions_taken = []
        else:
            actions_taken.append(action)
        update_obs_map(action, arr, obs_map, sub_x, sub_y,M,N)
        previous_step=action
        sub_index = np.where(obs_map == 6)
        sub_x, sub_y = (sub_index[0][0], sub_index[1][0])
        if len(state) == 3:
            state = (sub_x, sub_y, state[2])
        elif len(state) == 4:
            state = (sub_x, sub_y, state[2], state[3])
        elif len(state) == 5:
            state = (sub_x, sub_y, state[2], state[3], state[4])
        row_range = range(max(sub_x-1, 0), min(sub_x+1, len(arr)-1)+1)
        col_range = range(max(sub_y-1, 0), min(sub_y+1, len(arr[0])-1)+1)
        obs_treasure_index = np.where(obs_map == 2) # updates how many treasures have been seen
        num_treasures_left = len(obs_treasure_index[0])
        draw_map(screen, obs_map)

    # update state
    if len(treasure_index[0]) == 1:
            state = (sub_x, sub_y, 0)  # start state
    elif len(treasure_index[0]) == 2:
        state = (sub_x, sub_y, 0, 0)  # start state
    elif len(treasure_index[0]) == 3:
        state = (sub_x, sub_y, 0, 0, 0)

    # find obsereved treasure indexs
    obs_treasure = {}
    for i in range(len(obs_treasure_index[0])):
        index = (obs_treasure_index[0][i], obs_treasure_index[1][i])
        obs_treasure[i] = index

    # Obstacle locations
    obstacles_index = np.where((obs_map == 3) |(obs_map == 4) | (obs_map == 5) | (obs_map == 9))
    obstacles = {}
    for i in range(len(obstacles_index[0])):
        index = (obstacles_index[0][i], obstacles_index[1][i])
        obstacles[i] = index
    obstacle_values = obstacles.values()

    #retrieve treasure
    optimal_policy, optimal_values = policy_iteration(obstacle_values)
    plan, total_reward, state, num_treasures_left = execute_policy(optimal_policy, state, num_treasures_left)
    sub_x = state[0]
    sub_y = state[1]
    previous_step = 'return'

# Visualizing end screen with pygame
waves_sound.fadeout(2000)
game_sound.fadeout(2000)
size = (N * 50, M * 50)
screen = pygame.display.set_mode(size)
success_sound.set_volume(50)
success_sound.play()
screen.blit(end_image, (0, 0))
pygame.display.update()

pygame.time.delay(5000)

pygame.quit()
