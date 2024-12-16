import copy
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.utils.step_api_compatibility import step_api_compatibility


def print_buffers(obs_buffer, step):

    trajectory = []
    for observation in obs_buffer:
        observation = observation + 1 / 2
        observation = observation.detach().cpu().numpy()
        #observation = np.transpose(observation, (1, 2, 0))
        trajectory.append(observation)

    plot_images(trajectory, step)


def get_best_child(node):

    max_visits = -1
    best_child = None
    for key, child in node.children.items():
        if child.visit_count > max_visits:
            max_visits = child.visit_count
            best_child = child

    return best_child

def create_trajectory(node, start_step):

    trajectory = []

    while len(node.children) > 0:
        obs = get_observation(node)
        trajectory.append(obs)
        node = get_best_child(node)

    plot_images(trajectory[:16], start_step)

    return len(trajectory[:16])

def get_observation(node):
    observation = node.observation.squeeze()
    if observation.shape[0] == 3:  # Check if the channels are the first dimension
        observation = np.transpose(observation, (1, 2, 0))  # Rearrange to (64, 64, 3)
    return observation

def pad_images(images: np.ndarray, top=0, bottom=0, left=0, right=0, constant=0) -> np.ndarray:
    assert len(images.shape) == 4, "not a batch of images!"
    return np.pad(images, ((0, 0), (top, bottom), (left, right), (0, 0)), mode="constant", constant_values=constant)


def plot_images(images, start_step, num_steps, transpose, title=""):


    images = images[:num_steps]
    if transpose:
        images = [np.transpose(obs, (1, 2, 0)) for obs in images]

    empty = np.array(images[0].copy())
    empty.fill(0)

    cols = math.sqrt(num_steps)
    if math.floor(cols) < cols:
        cols = math.floor(cols) + 1
    else:
        cols = math.floor(cols)  # for some reason this is needed

    rows = math.ceil(num_steps / cols)

    images.extend(((cols * rows) - len(images)) * [empty])

    padded_images = pad_images(np.array(images), top=4, bottom=4, left=4, right=4)
    image_rows = []
    resize_factor = 1
    for i in range(rows):
        image_slice = padded_images[i * cols: (i + 1) * cols]
        image_row = np.concatenate(image_slice, 1)
        x, y, _ = image_row.shape
        image_row_resized = image_row[::resize_factor, ::resize_factor]
        image_rows.append(image_row_resized)

    image = np.concatenate(image_rows, 0)

    plt.imshow(image)
    plt.axis('off')  # Optional: Turn off the axis
    plt.title(f"{title} - Step {start_step}")
    plt.show()

def load_observations(path):
    observations_path = f"{path}/observations.pkl"
    with open(observations_path, 'rb') as f:
        observations = pickle.load(f)
        observations = [obs.squeeze().detach().cpu().numpy() for obs in observations]
        observations = [np.transpose(obs, (1, 2, 0)) for obs in observations]

    return observations

def load_actions(path):

    mcts_actions_path = f"{path}/mcts_actions.pkl"
    policy_actions_path = f"{path}/policy_actions.pkl"
    with open(mcts_actions_path, 'rb') as f:
        mcts_actions = pickle.load(f)

    with open(policy_actions_path, 'rb') as f:
        policy_actions = pickle.load(f)

    return mcts_actions, policy_actions

def load_roots(path, step):

    tree_path = f"{path}/mcts_tree_{step}.pkl"
    with open(tree_path, 'rb') as f:
        root_node = pickle.load(f)

    return root_node

def get_plan_for_actions(roots, actions):

    node = copy.deepcopy(roots)
    observations = []
    for a in actions:
        observation = get_observation(node)
        observations.append(observation)
        if a in node.children and node.children[a].visit_count > 0:
            node = node.children[a]
        else:
            print("Action not in children")
            break
    return observations


if __name__ == "__main__":

    step = 1120
    num_steps = 16
    file_path = f'mcts/iris'

    roots = load_roots(file_path, step)
    observations = load_observations(file_path)
    mcts_actions, policy_actions = load_actions(file_path)

    plot_images(observations[step:], step, num_steps, "Real Env")
    print("MCTS actions: ", mcts_actions[step:step+num_steps])
    print("Policy actions: ", policy_actions[step:step+num_steps])


    plan_observations = get_plan_for_actions(roots, mcts_actions[step:])
    plot_images(plan_observations, step, num_steps, "World Model")