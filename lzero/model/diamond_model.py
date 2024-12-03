"""
Overview:
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
import numpy as np
import torch.nn as nn
from omegaconf import open_dict
from torch.distributions import Categorical

from ding.utils import MODEL_REGISTRY
from numpy import ndarray

from diamond.src.models.diffusion import Denoiser, DiffusionSampler
from diamond.src.models.rew_end_model import RewEndModel
from diamond.src.models.actor_critic import ActorCritic
from diamond.src.utils import extract_state_dict
from iris.src.models.kv_caching import KeysValues
from zoo.atari.entry.tree_visualization import print_buffers
from .common import MZNetworkOutput, IrisNetworkOutput, DiamondNetworkOutput

from typing import Tuple

from hydra import compose, initialize

from pathlib import Path

import torch


@MODEL_REGISTRY.register('DiamondModel')
class DiamondModel(nn.Module):

    def __init__(
        self,  *args,
        **kwargs
    ):
        super().__init__()

        self.model_path = kwargs.get('model_path', None)
        self.env_id = kwargs.get('env_id', None)
        self.num_action = kwargs.get('action_space_size', None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler, self.rew_end_model, self.actor_critic = self.load_agent()
        self.true_model_hidden_state = (None, None)

    def load_agent(self) -> Tuple[DiffusionSampler, RewEndModel, ActorCritic]:

        path = "../../diamond/config/"
        with initialize(config_path=path, job_name="test_app"):
            cfg = compose(config_name="trainer")

        with open_dict(cfg):
            cfg.agent.denoiser.inner_model.num_actions = self.num_action
            cfg.agent.rew_end_model.num_actions = self.num_action
            cfg.agent.actor_critic.num_actions = self.num_action

        denoiser = Denoiser(cfg.agent.denoiser)
        rew_end_model = RewEndModel(cfg.agent.rew_end_model)
        actor_critic = ActorCritic(cfg.agent.actor_critic)

        device = torch.device(cfg.common.devices)
        sd = torch.load(Path(f"../../../{self.model_path}"), map_location=device)
        sd = {k: extract_state_dict(sd, k) for k in ("denoiser", "rew_end_model", "actor_critic")}
        denoiser.load_state_dict(sd["denoiser"])
        rew_end_model.load_state_dict(sd["rew_end_model"])
        actor_critic.load_state_dict(sd["actor_critic"])

        denoiser.to(device)
        rew_end_model.to(device)
        actor_critic.to(device)

        sampler = DiffusionSampler(denoiser, cfg.world_model_env.diffusion_sampler)
        return sampler, rew_end_model, actor_critic

    def predict(self,  obs: torch.Tensor, ac_hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        logits_actions, value, model_hidden_state = self.actor_critic.predict_act_value(obs, ac_hidden_state)
        return logits_actions, value, model_hidden_state

    def initial_inference(self, obs: torch.Tensor, ac_hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.predict(obs=obs, ac_hidden_state=ac_hidden_state)

    def predict_next_obs(self, obs_buffer, act_buffer):
        return self.sampler.sample(obs_buffer, act_buffer)

    def predict_rew_end(self, obs_buffer, act_buffer, next_obs, rew_end_hidden_state):

        logits_rew, logits_end, (hx_rew_end, cx_rew_end) = self.rew_end_model.predict_rew_end(
            obs_buffer[:, -1:], act_buffer[:, -1:], next_obs, rew_end_hidden_state)
        rew = Categorical(logits=logits_rew).sample().squeeze(1) - 1.0  # in {-1, 0, 1}
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end, (hx_rew_end, cx_rew_end)

    def recurrent_inference(self,
                            observation: torch.Tensor,
                            action: torch.Tensor,
                            ac_hidden_state: Tuple[torch.Tensor, torch.Tensor],
                            rew_end_hidden_state: Tuple[torch.Tensor, torch.Tensor],
                            obs_buffer: torch.Tensor,
                            act_buffer: torch.Tensor) -> DiamondNetworkOutput:


        act_buffer[:, -1] = action # shape [batch_size, num_stacked_frames] -> [1, 4]
        next_obs, _ = self.predict_next_obs(obs_buffer.unsqueeze(0).to(self.device), act_buffer.to(self.device))
        rew, end, rew_end_hidden_state = self.predict_rew_end(obs_buffer.unsqueeze(0), act_buffer, next_obs, rew_end_hidden_state)

        new_act_buffer = act_buffer.detach().clone()
        new_act_buffer = new_act_buffer.roll(-1, dims=1) # rotate the buffer to the left

        new_obs_buffer = obs_buffer.detach().clone()
        new_obs_buffer = new_obs_buffer.roll(-1, dims=0)
        new_obs_buffer[-1] = next_obs

        policy_logits, value, ac_hidden_state = self.predict(next_obs, ac_hidden_state=ac_hidden_state)

        output = DiamondNetworkOutput(
            value=value,
            reward=rew,
            policy_logits=policy_logits,
            observation=next_obs,
            ac_hidden_state=ac_hidden_state,
            rew_end_hidden_state=rew_end_hidden_state,
            act_buffer=new_act_buffer,
            obs_buffer=new_obs_buffer,
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
