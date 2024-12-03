import math
import pickle
import matplotlib.pyplot as plt
import numpy as np


def print_buffers(obs_buffer, step):

    trajectory = []
    for observation in obs_buffer:
        observation = observation + 1 / 2
        observation = observation.detach().cpu().numpy()
        observation = np.transpose(observation, (1, 2, 0))
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

def get_observation(node):
    observation = node.observation.squeeze()
    if observation.shape[0] == 3:  # Check if the channels are the first dimension
        observation = np.transpose(observation, (1, 2, 0))  # Rearrange to (64, 64, 3)
    return observation

def pad_images(images: np.ndarray, top=0, bottom=0, left=0, right=0, constant=0) -> np.ndarray:
    assert len(images.shape) == 4, "not a batch of images!"
    return np.pad(images, ((0, 0), (top, bottom), (left, right), (0, 0)), mode="constant", constant_values=constant)


def plot_images(images, start_step):
    image_len = len(images)

    empty = np.array(images[0].copy())
    empty.fill(0)

    cols = math.sqrt(image_len)
    if math.floor(cols) < cols:
        cols = math.floor(cols) + 1
    else:
        cols = math.floor(cols)  # for some reason this is needed

    rows = math.ceil(len(images) / cols)

    images.extend(((cols * rows) - image_len) * [empty])

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
    plt.title(f'Observation {start_step}')
    plt.show()


if __name__ == "__main__":

    step = 438
    # Assuming you saved the object in this file
    file_path = f'mcts/mcts_tree_{step}.pkl'

    # Load the object
    with open(file_path, 'rb') as f:
        root_node = pickle.load(f)

    create_trajectory(root_node, step)

