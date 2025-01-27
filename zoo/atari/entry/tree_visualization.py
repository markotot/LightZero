import copy
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np


def print_buffers(obs_buffer, step):

    trajectory = []
    for observation in obs_buffer:
        observation = observation.detach().cpu().numpy()
        # observation = observation + 1 / 2
        # observation = np.transpose(observation, (1, 2, 0))
        trajectory.append(observation)

    plot_images(trajectory, step, num_steps=len(trajectory), transpose=True)


def get_best_child(node):

    max_visits = -1
    best_child = None
    for key, child in node.children.items():
        if child.visit_count > max_visits:
            max_visits = child.visit_count
            best_child = child

    return best_child

def create_trajectory(node, num_steps):

    trajectory = []
    actions = []
    while len(node.children) > 0:
        obs = get_observation(node)
        trajectory.append(obs)
        actions.append(node.best_action)
        node = get_best_child(node)

    return trajectory[:num_steps], actions[:num_steps]

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

    plt.figure(dpi=300)
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


def breakout_action_to_str(action):
    if action == 0:
        return "NOOP"
    elif action == 1:
        return "FIRE"
    elif action == 2:
        return "RIGHT"
    elif action == 3:
        return "LEFT"


if __name__ == "__main__":

    step = 560
    num_steps = 16
    algorithm = "diamond"
    file_path = f'mcts/{algorithm}'

    roots = load_roots(file_path, step)
    observations = load_observations(file_path)
    mcts_actions, policy_actions = load_actions(file_path)

    plan_observations = get_plan_for_actions(roots, mcts_actions[step:])
    planned_traj, planned_actions = create_trajectory(roots, step)

    if algorithm == "diamond":
        plan_observations = [(obs + 1) / 2 for obs in plan_observations]
        planned_traj = [(obs + 1) / 2 for obs in planned_traj]


    mcts_str_actions = [breakout_action_to_str(x) for x in mcts_actions[step:step + num_steps]]
    policy_str_actions = [breakout_action_to_str(x) for x in policy_actions[step:step + num_steps]]
    planned_str_actions = [breakout_action_to_str(x) for x in planned_actions]
    print("MCTS actions:\t", mcts_str_actions)
    print("Policy actions:\t", policy_str_actions)
    print("Planned actions:\t", planned_str_actions)

    plot_images(observations[step:], step, num_steps, transpose=False, title="Real Env")
    plot_images(plan_observations, step, num_steps, transpose=False, title=f"{algorithm}: WM Same Actions")
    plot_images(planned_traj, step, num_steps, transpose=False, title=f"{algorithm}: Best Planned Trajectory")
    print("Hi!")