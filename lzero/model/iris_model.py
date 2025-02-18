"""
Overview:
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical

from ding.utils import MODEL_REGISTRY
from numpy import ndarray

from iris.src.models.kv_caching import KeysValues
from zoo.atari.entry.tree_visualization import plot_images
from .common import MZNetworkOutput, IrisNetworkOutput

from typing import Tuple, Any

from hydra import compose, initialize
from wandb.wandb_agent import Agent

from functools import partial
from pathlib import Path

from hydra.utils import instantiate
import torch

from iris.src.agent import Agent
from iris.src.envs.single_process_env import SingleProcessEnv
from iris.src.envs.world_model_env import WorldModelEnv
from iris.src.models.actor_critic import ActorCritic
from iris.src.models.world_model import WorldModel

import psutil
import os

import torch.optim as optim

@MODEL_REGISTRY.register('IrisModel')
class IrisModel(nn.Module):

    def __init__(
        self,  *args,
        **kwargs
    ):
        super().__init__()

        self.model_path = kwargs.get('model_path', None)
        self.env_id = kwargs.get('env_id', None)

        self.config_path = kwargs.get('model_cfg', None)
        self.agent, self.world_model_env = self.load_agent(config_path=self.config_path)
        self.true_model_hidden_state = (None, None)

    def load_agent(self, config_path):
        with initialize(config_path=config_path, job_name="test_app"):
            cfg = compose(config_name="trainer")

        device = torch.device(cfg.common.device)
        assert cfg.mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

        cfg.env.test.id = self.env_id # override iris config to the LightZero config
        env_fn = partial(instantiate, config=cfg.env.test)
        test_env = SingleProcessEnv(env_fn)

        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions,
                                 config=instantiate(cfg.world_model))

        #world_model = torch.compile(world_model, mode='default')
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=test_env.num_actions)
        actor_critic.reset(1)
        agent = Agent(tokenizer, world_model, actor_critic).to(device)

        agent.load(Path(f"../../../{self.model_path}"), device)

        world_model_env = WorldModelEnv(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device, env=env_fn())
        return agent, world_model_env


    def predict(self,  obs: torch.Tensor, model_hidden_state: Tuple[np.ndarray, np.ndarray], temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_ac = obs if self.agent.actor_critic.use_original_obs else torch.clamp(
            self.agent.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        self.agent.actor_critic.set_hidden_state(model_hidden_state)
        ac_output = self.agent.actor_critic(inputs=input_ac, mask_padding=None)
        logits_actions = ac_output.logits_actions[:, -1] / temperature
        return logits_actions, ac_output.means_values, self.agent.get_model_hidden_state()

    def initial_inference(self, obs: torch.Tensor, model_hidden_state: Tuple[np.ndarray, np.ndarray],) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        policy_logits, value, hidden_state = self.predict(obs=obs, model_hidden_state=model_hidden_state, temperature=1.0)
        return policy_logits, value, hidden_state

    def recurrent_inference(self,
                            model_hidden_state: Tuple[np.ndarray, np.ndarray],
                            obs_seq: torch.Tensor,
                            tokens_seq: torch.Tensor,
                            action_seq: torch.Tensor) -> IrisNetworkOutput:

        next_obs, next_tokens, reward, done, _ = self.world_model_env.step_with_tokens(action_seq, tokens_seq)
        # next_obs, reward, done, _ = self.world_model_env.step_v2(action_seq, obs_seq) # TODO: enable this for pixel reconstruction


        # last_env_obs = obs_seq[-1][0].detach().cpu().numpy()
        # wm_obs = next_obs[0].detach().cpu().numpy()
        # plot_images([last_env_obs, wm_obs], start_step=0, num_steps=2, transpose=True)
        policy_logits, value, hidden_state = self.predict(next_obs, model_hidden_state=model_hidden_state, temperature=1.0)

        ac_hidden_state = (hidden_state[0].detach().cpu(), hidden_state[1].detach().cpu())

        output = IrisNetworkOutput(
            value=value, #TODO: get real value, maybe from AC model?
            reward=torch.from_numpy(reward),
            policy_logits=policy_logits,
            observation=next_obs,
            tokens=next_tokens,
            ac_hidden_state=ac_hidden_state,
        )

        return output

    def _representation(self, observation: torch.Tensor) -> torch.Tensor:
        pass

    def _prediction(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def _dynamics(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def project(self, latent_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        pass


class DynamicsNetwork(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state_action_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_dynamic_mean(self) -> float:
        pass

    def get_reward_mean(self) -> Tuple[ndarray, float]:
        pass
