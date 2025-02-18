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
        self._num_observations_tokens = world_model.config.tokens_per_block - 1
        self._seq_length = world_model.config.max_blocks

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def step_v2(self, actions, observations):

        observations = torch.cat(observations, dim=0).to(self.device)
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens
        act_tokens = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)
        obs_act_sequence = torch.cat((obs_tokens, act_tokens), dim=1)
        obs_act_sequence = obs_act_sequence.flatten().unsqueeze(0)
        max_length = self._seq_length * (self._num_observations_tokens + 1)

        for k in range(self._num_observations_tokens):

            input_tokens = obs_act_sequence[:, -max_length+self._num_observations_tokens + 1 - k:] if obs_act_sequence.size(1) >= -max_length else obs_act_sequence
            outputs_wm = self.world_model(input_tokens, past_keys_values=self.keys_values_wm)

            # First forward pass after sending action token provides reward and done
            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)

            next_token = Categorical(logits=outputs_wm.logits_observations[:,-1]).sample()
            obs_act_sequence = torch.cat((obs_act_sequence, next_token.unsqueeze(0)), dim=1)

        self.obs_tokens = obs_act_sequence[:, -self._num_observations_tokens:]
        obs = self.decode_obs_tokens()
        return obs, reward, done, None

    @torch.no_grad()
    def step_with_tokens(self, actions, tokens):

        obs_tokens = torch.cat(tokens, dim=0).to(self.device)
        # obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens
        act_tokens = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)
        obs_act_sequence = torch.cat((obs_tokens, act_tokens), dim=1)
        obs_act_sequence = obs_act_sequence.flatten().unsqueeze(0)
        max_length = self._seq_length * (self._num_observations_tokens + 1)

        for k in range(self._num_observations_tokens):

            input_tokens = obs_act_sequence[:, -max_length + self._num_observations_tokens + 1 - k:] if obs_act_sequence.size(
                1) >= -max_length else obs_act_sequence
            outputs_wm = self.world_model(input_tokens, past_keys_values=self.keys_values_wm)

            # First forward pass after sending action token provides reward and done
            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)

            next_token = Categorical(logits=outputs_wm.logits_observations[:, -1]).sample()
            obs_act_sequence = torch.cat((obs_act_sequence, next_token.unsqueeze(0)), dim=1)

        self.obs_tokens = obs_act_sequence[:, -self._num_observations_tokens:]
        obs = self.decode_obs_tokens()
        return obs, self.obs_tokens, reward, done, None

    # Returns observations and observation tokens
    @torch.no_grad()
    def step_return_tokens(self, actions, observations):

        observations = torch.cat(observations, dim=0).to(self.device)
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens
        act_tokens = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)
        obs_act_sequence = torch.cat((obs_tokens, act_tokens), dim=1)
        obs_act_sequence = obs_act_sequence.flatten().unsqueeze(0)
        max_length = self._seq_length * (self._num_observations_tokens + 1)

        for k in range(self._num_observations_tokens):

            input_tokens = obs_act_sequence[:, -max_length + self._num_observations_tokens + 1 - k:] if obs_act_sequence.size(
                1) >= -max_length else obs_act_sequence
            outputs_wm = self.world_model(input_tokens, past_keys_values=self.keys_values_wm)

            next_token = Categorical(logits=outputs_wm.logits_observations[:, -1]).sample()
            obs_act_sequence = torch.cat((obs_act_sequence, next_token.unsqueeze(0)), dim=1)

        self.obs_tokens = obs_act_sequence[:, -self._num_observations_tokens:]
        obs = self.decode_obs_tokens()

        return obs, self.obs_tokens


    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

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
    def vq_encoder_only(self, observations) -> None:

        obs_tokens = self.encode_obs(observations)
        embedded_tokens = self.tokenizer.embedding(obs_tokens)  # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        decoded_obs = self.tokenizer.decode(z, should_postprocess=True)  # (B, C, H, W)
        decoded_obs = torch.clamp(decoded_obs, 0, 1)
        return decoded_obs, obs_tokens

    @torch.no_grad
    def encode_obs(self, observations):
        observations = torch.cat(observations, dim=0).to(self.device)
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens
        return obs_tokens

    @torch.no_grad
    def decode_tokens(self, obs_tokens):
        embedded_tokens = self.tokenizer.embedding(obs_tokens)  # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        decoded_obs = self.tokenizer.decode(z, should_postprocess=True)  # (B, C, H, W)
        decoded_obs = torch.clamp(decoded_obs, 0, 1)
        return decoded_obs

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