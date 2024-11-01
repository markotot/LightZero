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
        self.agent, self.world_model_env = self.load_agent()
        self.true_model_hidden_state = (None, None)

    def load_agent(self):
        path = "../../iris/config/"
        with initialize(config_path=path, job_name="test_app"):
            cfg = compose(config_name="trainer")

        device = torch.device(cfg.common.device)
        assert cfg.mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

        env_fn = partial(instantiate, config=cfg.env.test)
        test_env = SingleProcessEnv(env_fn)

        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions,
                                 config=instantiate(cfg.world_model))

        #world_model = torch.compile(world_model, mode='default')
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=test_env.num_actions)
        actor_critic.reset(1)
        agent = Agent(tokenizer, world_model, actor_critic).to(device)

        agent.load(Path('../../../iris/checkpoints/iris/Breakout.pt'), device)

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
                            observation: torch.Tensor,
                            action: torch.Tensor,
                            model_hidden_state: Tuple[np.ndarray, np.ndarray],
                            wm_keys_values: KeysValues) -> IrisNetworkOutput:

        print("Start recurrent_inference")
        print(f"Memory used script: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")

        if wm_keys_values is None:
            next_obs = self.world_model_env.reset_from_initial_observations(observation)
        else:
            self.world_model_env.set_kv_cache(wm_keys_values)

        print(f"Memory used script: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")


        next_obs, reward, done, _ = self.world_model_env.step(action, should_predict_next_obs=True)
        policy_logits, value, hidden_state = self.predict(next_obs, model_hidden_state=model_hidden_state, temperature=1.0)

        print(f"Memory used script: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")


        output = IrisNetworkOutput(
            value=value, #TODO: get real value, maybe from AC model?
            reward=torch.from_numpy(reward),
            policy_logits=policy_logits,
            latent_state=next_obs,
            hidden_state=hidden_state,
        )

        print(f"Memory used script: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
        print("End recurrent_inference")

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
