import copy
from functools import partial
from pathlib import Path

import hydra
import numpy as np
from Cython.Compiler.PyrexTypes import modifiers_and_name_to_type
from hydra import initialize, compose
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import DictConfig, open_dict
import torch

from diamond.src.models.diffusion import Denoiser as DiamondDenoiser
from diamond.src.models.actor_critic import ActorCritic as DiamondActorCritic
from diamond.src.models.diffusion import DiffusionSampler as DiamondDiffusionSampler
from diamond.src.models.rew_end_model import RewEndModel as DiamondRewEndModel

from diamond.src.utils import extract_state_dict
from iris.src.agent import Agent
from iris.src.envs.single_process_env import SingleProcessEnv

from iris.src.envs.world_model_env import WorldModelEnv as IrisWorldModelEnv
from iris.src.models.actor_critic import ActorCritic as IrisActorCritic
from iris.src.models.world_model import WorldModel as IrisWorldModel

import zoo.atari.config.atari_diamond_model_config as diamond_config
import zoo.atari.config.atari_iris_model_config as iris_config
from ding.config import compile_config
from ding.envs import create_env_manager, get_vec_env_setting
from ding.utils import set_pkg_seed

from zoo.atari.entry.tree_visualization import plot_images, breakout_action_to_str

def setup_env(seed, env_id):
    main_config, create_config = iris_config.get_configs(env_id)

    cfg = compile_config(main_config, seed=seed, env=None, auto=True, create_cfg=create_config, save_cfg=True)

    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    return env, cfg

def setup_iris(env_id, device):

    path = "../../../iris/config/"
    with initialize(config_path=path, job_name="test_app"):
        cfg = compose(config_name="trainer")

    cfg.env.test.id = env_id  # override iris config to the LightZero config
    model_path = iris_config.get_model_path_from_env_id(env_id)

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    tokenizer = instantiate(cfg.tokenizer)
    world_model = IrisWorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions,
                             config=instantiate(cfg.world_model))

    actor_critic = IrisActorCritic(**cfg.actor_critic, act_vocab_size=test_env.num_actions)
    actor_critic.reset(1)
    agent = Agent(tokenizer, world_model, actor_critic).to(device)

    # so many ../.. because we are in "./LightZero/zoo/atari/entry/outputs/2024-12-19/12-39-16"
    agent.load(Path(f"../../../{model_path}"), device)

    world_model_env = IrisWorldModelEnv(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                    env=env_fn())
    return agent, world_model_env

def setup_diamond(env_id, num_actions, device):

    path = "../../../diamond/config/"
    with initialize(config_path=path, job_name="test_app"):
        cfg = compose(config_name="trainer")

    cfg.env.test.id = env_id  # override iris config to the LightZero config
    model_path = diamond_config.get_model_path_from_env_id(env_id)

    with open_dict(cfg):
        cfg.agent.denoiser.inner_model.num_actions = num_actions
        cfg.agent.rew_end_model.num_actions = num_actions
        cfg.agent.actor_critic.num_actions = num_actions

    denoiser = DiamondDenoiser(cfg.agent.denoiser)
    rew_end_model = DiamondRewEndModel(cfg.agent.rew_end_model)
    actor_critic = DiamondActorCritic(cfg.agent.actor_critic)

    sd = torch.load(Path(f"../../../{model_path}"), map_location=device)
    sd = {k: extract_state_dict(sd, k) for k in ("denoiser", "rew_end_model", "actor_critic")}
    denoiser.load_state_dict(sd["denoiser"])
    rew_end_model.load_state_dict(sd["rew_end_model"])
    actor_critic.load_state_dict(sd["actor_critic"])

    denoiser.to(device)
    rew_end_model.to(device)
    actor_critic.to(device)

    sampler = DiamondDiffusionSampler(denoiser, cfg.world_model_env.diffusion_sampler)
    return sampler, rew_end_model, actor_critic

def get_obs_act_seq(observations, actions, start_step, device):
    observations = [torch.from_numpy(np.expand_dims(obs, axis=0)).to(device) for obs in observations[start_step: start_step + 20]]
    actions = actions[start_step: start_step + 20]
    return observations, actions

def  run_gym_env(env, max_steps, agent, device):

    env.launch()
    env.reset()

    timestep = env.ready_obs[0]

    obs = timestep['observation']
    env_observations = [obs]
    actions = []
    total_reward = 0
    # Generate a trajectory with real env
    num_steps = 0

    initial_lives = 5
    lost_lives_timesteps = []
    while num_steps < max_steps:

        obs_tensor = torch.from_numpy(obs).to(device)
        ac_output = agent.act(obs_tensor.unsqueeze(0), should_sample=False, temperature=1)
        action = ac_output.detach().cpu().numpy()[0]
        actions_dict = {0: action}

        obs_dict, rew, done, info = env.step(actions_dict)[0]
        obs = obs_dict['observation']
        env_observations.append(obs)
        actions.append(action)

        total_reward += rew
        if info['lives'] < initial_lives:
            lost_lives_timesteps.append(num_steps)
            initial_lives = info['lives']
        num_steps += 1
        if num_steps % 1000 == 0:
            print(num_steps)
        if done:
            print(num_steps)
            break

    return env_observations, actions, lost_lives_timesteps

def run_iris_agent(observations, actions, world_model_env, start_step, device):

    generated_steps = 12
    obs_seq, act_seq = get_obs_act_seq(observations, actions, start_step, device)
    wm_obs_seq = []
    for i in range(generated_steps):
        wm_input_actions = actions[start_step + i: start_step + i + 20]
        wm_input_obs = obs_seq[i: i + 20] + wm_obs_seq

        wm_obs, _, _, _ = world_model_env.step_v2(wm_input_actions, wm_input_obs)
        wm_obs_seq.append(wm_obs)

    num_plotted_images = 16
    generation_start = start_step + 20
    end_step = generation_start + generated_steps

    # Plot 16 images from environment
    plot_images(observations[end_step - num_plotted_images:end_step], generation_start, num_plotted_images,
                transpose=True, title="Iris-Gym-Env")

    # Plot 16 images, the first four from the environment, last 12 from the world model
    wm_observations = [obs.squeeze().detach().cpu().numpy() for obs in wm_obs_seq]
    plot_images(
        observations[generation_start + generated_steps - num_plotted_images: generation_start] + wm_observations,
        generation_start, num_plotted_images, transpose=True, title="Iris-WM")

def run_diamond_agent(observations, actions, diamond_wm, diamond_rew_end, start_step, device):


    generated_steps = 12
    wm_obs_seq = []
    for i in range(generated_steps):

        buffer_obs = [torch.from_numpy(obs * 2 - 1).to(device) for obs in observations[start_step + i: start_step + 4]]
        if i > 0:
            buffer_obs += wm_obs_seq[-4:]

        obs_buffer = torch.stack(buffer_obs)
        act_buffer = torch.tensor(np.stack(actions[start_step + i: start_step + i + 4]), device=device)
        wm_obs = diamond_wm.sample(obs_buffer.unsqueeze(0), act_buffer.unsqueeze(0))
        wm_obs_seq.append(wm_obs[0].squeeze())

    num_plotted_images = 16
    generation_start = start_step + 20

    # Plot 16 images from environment
    plot_images(observations[start_step: start_step + num_plotted_images], generation_start, num_plotted_images,
                transpose=True, title="Diamond-Gym-Env")

    # Plot 16 images, the first four from the environment, last 12 from the world model
    wm_observations = [(obs.squeeze().detach().cpu().numpy() + 1) / 2 for obs in wm_obs_seq]
    plot_images(
        observations[start_step: start_step + 4] + wm_observations,
        generation_start, num_plotted_images, transpose=True, title="Diamond-WM")

def burn_in_iris_test(observations, actions, iris_wm_env, start_step, device):

    num_decoded_obs = 12
    observation_slice = observations[start_step: start_step + num_decoded_obs]

    # Plot original images
    original_obs = copy.deepcopy(observation_slice)
    plot_images(original_obs, start_step, num_decoded_obs, transpose=True, title="Gym-Env-Original")

    # Plot decoded original images
    tensor_observations = [torch.from_numpy(np.expand_dims(obs, axis=0)).to(device) for obs in original_obs]
    decoded_obs, encoded_tokens = iris_wm_env.vq_encoder_only(tensor_observations)
    decoded_obs = [obs.squeeze().detach().cpu().numpy() for obs in decoded_obs]
    plot_images(decoded_obs, start_step, num_decoded_obs, transpose=True, title="Iris-WM-Original")

    # Plot modified images
    modified_observations = [modify_observation(x) for x in observation_slice]
    plot_images(modified_observations, start_step, num_decoded_obs, transpose=True, title="Iris-Env-Modified")

    # Plot decoded modified images
    tensor_observations = [torch.from_numpy(np.expand_dims(obs, axis=0)).to(device) for obs in modified_observations]
    decoded_obs, encoded_tokens = iris_wm_env.vq_encoder_only(tensor_observations)
    decoded_obs = [obs.squeeze().detach().cpu().numpy() for obs in decoded_obs]
    plot_images(decoded_obs, start_step, num_decoded_obs, transpose=True, title="Iris-WM-Modified")

def tokens_saturation(observations, iris_wm_env,device):


    tensor_observations = [torch.from_numpy(np.expand_dims(obs, axis=0)).to(device) for obs in observations]
    decoded_obs, encoded_tokens = iris_wm_env.vq_encoder_only(tensor_observations)

    histogram = torch.histc(encoded_tokens, bins=512, min=0, max=511)
    histogram = histogram.detach().cpu().numpy()
    plt.bar(range(512), histogram, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Histogram of Tensor Values")
    plt.xlabel("Bin")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return histogram


def modify_observation(obs):

    # Horizontal line
    obs[:, 20:30, :] = 0

    # Vertical line
    #obs[:, :, 32] = 0
    return obs

def main():

    start_step = 50
    seed = 0
    env_id = "BreakoutNoFrameskip-v4"
    max_steps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env, light_zero_cfg = setup_env(seed, env_id)
    iris_agent, iris_wm_env = setup_iris(env_id, device)
    diamond_wm_env, diamond_rew_end, diamond_actor_critic = setup_diamond(env_id, num_actions=light_zero_cfg.policy.model.action_space_size, device=device)

    print("Starting the envs")
    observations, actions, lost_lives_timesteps = run_gym_env(env, max_steps, iris_agent, device)
    print(lost_lives_timesteps)


    run_iris_agent(observations, actions, iris_wm_env, start_step, device)

    histogram = tokens_saturation(observations, iris_wm_env, device)
    burn_in_iris_test(observations, actions, iris_wm_env, start_step, device)
    # run_diamond_agent(observations, actions, diamond_wm_env, diamond_rew_end, start_step, device)
    #
    # action_strings = [breakout_action_to_str(action) for action in actions]
    # print(action_strings[start_step: start_step + 16])

if __name__ == "__main__":

    main()
