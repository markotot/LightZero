from functools import partial
from pathlib import Path

import hydra
from hydra.utils import instantiate
import omegaconf
import torch

from src.agent import Agent
from src.envs import SingleProcessEnv
from src.models.actor_critic import ActorCritic
from src.models.world_model import WorldModel


from hydra import compose, initialize
from omegaconf import OmegaConf


def main():

    # Load the hydra config without using hydra main
    with initialize(config_path="./config", job_name="test_app"):
        cfg = compose(config_name="trainer")

    device = torch.device(cfg.common.device)
    assert cfg.mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    tokenizer = instantiate(cfg.tokenizer)
    world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions,
                             config=instantiate(cfg.world_model))
    actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=test_env.num_actions)
    agent = Agent(tokenizer, world_model, actor_critic).to(device)
    agent.load(Path('../../../checkpoints/iris/Breakout.pt'), device)

    print(agent)


if __name__ == "__main__":
    main()
