import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 
import termtables as tt 

from utils.format import cell_type_to_char

# 0 is right, 1 is down, 2 is left, 3 is up
DIR_TO_ARROW = {0: "\u2192", 1: "\u2193", 2: "\u2190", 3: "\u2191"}

def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env

def plot_env(env, name="plotted env"):
    img = env.render()

    # Plot the rendered image
    plt.imshow(img)
    plt.title(f"Phase {name}")

    plt.xticks([])
    plt.yticks([])

    plt.show()

def env_to_str(env, sequence=False):
    grid_list = []

    for cell in env.grid.grid:
        try:
            grid_list.append(cell.type)
        except AttributeError:
            grid_list.append("floor")

    # calculate the side length of the square array
    n = int(np.sqrt(len(grid_list)))

    grid_arr = np.array(list(map(cell_type_to_char, grid_list))).reshape(n, n)
    
    # place agent, need to switch coords from minigrid to numpy notation
    if sequence: #TODO: inaccurate but goo enough for now 
        grid_arr[env.agent_pos[1], env.agent_pos[0]] = "Y"
    else: 
        grid_arr[env.agent_pos[1], env.agent_pos[0]] = DIR_TO_ARROW[env.agent_dir]

    grid_str = tt.to_string(
        grid_arr,
        style=tt.styles.ascii_thin,
    )

    return grid_str