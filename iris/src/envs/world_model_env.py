import copy
import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

        self.obs_tokens = None
        self._num_observations_tokens = 16

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    # @torch.no_grad()
    # def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
    #     obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
    #     _, num_observations_tokens = obs_tokens.shape
    #     if self.num_observations_tokens is None:
    #         self._num_observations_tokens = num_observations_tokens
    #
    #     _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
    #     self.obs_tokens = obs_tokens
    #
    #     return self.decode_obs_tokens()
    #
    # @torch.no_grad()
    # def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
    #     n, num_observations_tokens = obs_tokens.shape
    #     assert num_observations_tokens == self.num_observations_tokens
    #     self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
    #     outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
    #     return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def step_v2(self, actions, observations):


        observations = torch.cat(observations, dim=0).to(self.device)
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens
        act_tokens = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)
        obs_act_sequence = torch.cat((obs_tokens, act_tokens), dim=1)
        obs_act_sequence = obs_act_sequence.flatten().unsqueeze(0)
        num_passes = 16 # assumed that number of tokens for observation is 16
        max_length = 340

        for k in range(num_passes):

            input_tokens = obs_act_sequence[:, -max_length+num_passes + 1 - k:] if obs_act_sequence.size(1) >= -max_length else obs_act_sequence
            outputs_wm = self.world_model(input_tokens, past_keys_values=self.keys_values_wm)

            # First forward pass after sending action token provides reward and done
            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)

            next_token = Categorical(logits=outputs_wm.logits_observations[:,-1]).sample()
            obs_act_sequence = torch.cat((obs_act_sequence, next_token.unsqueeze(0)), dim=1)

        self.obs_tokens = obs_act_sequence[:, -num_passes:]
        obs = self.decode_obs_tokens()
        return obs, reward, done, None

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        # if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
        #     _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)


        for k in range(num_passes):  # assumption that there is only one action token.
            print(f"Step {k}, Token: {token}")
            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)
        obs = self.decode_obs_tokens() if should_predict_next_obs else None

        return obs, reward, done, None

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]

    def get_a_copy_of_kv_cache(self):
        x = copy.deepcopy(self.keys_values_wm)
        x.to_device('cpu')
        return x
    #
    def set_kv_cache(self, kv_cache):
        kv_cache = copy.deepcopy(kv_cache)
        kv_cache.to_device(self.device)
        self.keys_values_wm = kv_cache

    # def get_a_copy_of_kv_cache(self):
    #     x = self.keys_values_wm.to_numpy()
    #     return x
    #
    # def set_kv_cache(self, kv_cache):
    #     self.keys_values_wm.to_tensor(kv_cache, self.device)